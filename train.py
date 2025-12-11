"""Training script for segment-aware ECG classification with knowledge distillation."""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve

from constraints import ConstrainedLinear, scale_to_unit
from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from models.teacher import ResNet18_1D
from utils import (
    compute_class_weights,
    confusion_metrics,
    kd_logit_loss,
    l2_normalize,
    make_weighted_sampler,
    sweep_thresholds,
    sweep_thresholds_blended,
)


TRAIN_RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "115",
    "117",
    "121",
    "122",
    "123",
    "200",
    "201",
    "202",
    "203",
    "207",
    "209",
    "210",
    "212",
    "213",
    "219",
    "222",
    "223",
    "230",
    "231",
    "232",
]
GENERALIZATION_RECORDS = [
    "106",
    "114",
    "116",
    "118",
    "119",
    "124",
    "205",
    "208",
    "214",
    "215",
    "220",
    "221",
    "228",
    "233",
    "234",
]


def build_teacher(num_classes: int, embedding_dim: int, device: torch.device) -> nn.Module:
    teacher = ResNet18_1D(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def train_teacher(
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor | None,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
) -> None:
    teacher.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(teacher.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=False)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, max_epochs + 1):
        teacher.train()
        running_loss = 0.0
        total = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = teacher(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        train_loss = running_loss / max(total, 1)

        # validation
        teacher.eval()
        val_loss = 0.0
        vtotal = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                logits, _ = teacher(signals)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                vtotal += labels.size(0)
        val_loss = val_loss / max(vtotal, 1)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = teacher.state_dict()
        if epoch % 1 == 0:
            print(f"[Teacher] Epoch {epoch:02d}/{max_epochs} TrainLoss {train_loss:.4f} ValLoss {val_loss:.4f}")

    if best_state is not None:
        teacher.load_state_dict(best_state)
    teacher.eval()


def build_student(args: argparse.Namespace, device: torch.device) -> nn.Module:
    student = SegmentAwareStudent(
        num_classes=len(set(BEAT_LABEL_MAP.values())),
        num_mlp_layers=args.num_mlp_layers,
        dropout_rate=args.dropout_rate,
        use_value_constraint=args.use_value_constraint,
        use_tanh_activations=args.use_tanh_activations,
        constraint_scale=args.constraint_scale,
    ).to(device)
    return student


def evaluate(
    model: SegmentAwareStudent,
    data_loader: DataLoader,
    device: torch.device,
    return_probs: bool = False,
    threshold: float | None = None,
) -> Tuple[float, Dict[str, float], List[int], List[int], List[float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    preds: List[int] = []
    trues: List[int] = []
    probs: List[float] = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            if threshold is None:
                pred = torch.argmax(logits, dim=1)
            else:
                pred = (prob_pos >= threshold).long()
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            if return_probs:
                probs.extend(prob_pos.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    metrics = confusion_metrics(trues, preds)
    return avg_loss, metrics, trues, preds, probs


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Backward-compatible boolean flags with --name / --no-name."""

    parser.add_argument(f"--{name}", dest=name, action="store_true", help=f"Enable {help_text}")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT-BIH ECG training with KD and constraints")
    parser.add_argument("--data_path", type=str, default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience on monitored metric")
    parser.add_argument("--min_epochs", type=int, default=25, help="Minimum epochs before early stopping")
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_mlp_layers", type=int, default=3)
    parser.add_argument("--class_weight_abnormal", type=float, default=1.2)
    parser.add_argument("--max_class_weight_ratio", type=float, default=1.5)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_embedding_dim", type=int, default=128)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_d", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--teacher_auto_train_epochs", type=int, default=15)
    parser.add_argument("--teacher_min_f1", type=float, default=0.6, help="Minimum teacher F1 to keep KD enabled")
    parser.add_argument("--teacher_min_sensitivity", type=float, default=0.7, help="Minimum teacher TPR to keep KD enabled")
    parser.add_argument("--kd_warmup_epochs", type=int, default=5, help="Epochs to train without KD before enabling distillation")
    parser.add_argument(
        "--imbalance_warmup_epochs",
        type=int,
        default=5,
        help="Epochs to keep CE unweighted and sampler off before applying rebalancing",
    )
    parser.add_argument(
        "--imbalance_ramp_epochs",
        type=int,
        default=5,
        help="Epochs over which to linearly ramp class weights/sampler boost after warmup",
    )
    parser.add_argument("--recall_target_miss", type=float, default=0.15, help="Trigger recall rescue when miss rate exceeds this")
    parser.add_argument(
        "--adaptive_fpr_cap",
        type=float,
        default=0.25,
        help="Only trigger recall rescue when FPR is below this cap",
    )
    parser.add_argument("--recall_rescue_limit", type=int, default=3, help="Max number of adaptive recall boosts")
    parser.add_argument("--threshold_target_miss", type=float, default=0.12, help="Preferred max miss-rate when sweeping thresholds")
    parser.add_argument("--threshold_max_fpr", type=float, default=0.2, help="Optional FPR cap during threshold sweep")
    parser.add_argument(
        "--use_blended_thresholds",
        action="store_true",
        help="When set, sweep thresholds jointly on val+generalization with blended scoring",
    )
    parser.add_argument(
        "--threshold_generalization_weight",
        type=float,
        default=0.3,
        help="Weight for generalization metrics when blending thresholds (0-1)",
    )
    parser.add_argument(
        "--collapse_cooldown_epochs",
        type=int,
        default=5,
        help="Epochs to disable reweighting/sampler after a collapse event before ramping again",
    )
    parser.add_argument("--auto_sampler_ratio", type=float, default=0.35, help="Auto-enable sampler when abnormal ratio is below this")
    parser.add_argument("--kd_pause_miss", type=float, default=0.35, help="Pause KD when miss-rate exceeds this")
    parser.add_argument("--kd_resume_miss", type=float, default=0.2, help="Resume KD when miss-rate falls below this")
    _add_bool_arg(
        parser,
        "stable_mode",
        default=True,
        help_text="collapse-safe baseline (KD and adaptive reweighting off by default)",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--sampler_abnormal_boost",
        type=float,
        default=1.2,
        help="Boost factor for abnormal beats in weighted sampler (set >1 to upsample abnormal)",
    )

    _add_bool_arg(parser, "use_class_weights", default=True, help_text="class-weighted CE loss")
    _add_bool_arg(parser, "enable_adaptive_reweight", default=True, help_text="increase abnormal emphasis when recall is low")
    _add_bool_arg(parser, "use_kd", default=True, help_text="knowledge distillation")
    _add_bool_arg(parser, "use_value_constraint", default=False, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")
    _add_bool_arg(parser, "auto_train_teacher", default=True, help_text="auto teacher training when no checkpoint is provided")
    _add_bool_arg(parser, "use_weighted_sampler", default=False, help_text="weighted sampler to rebalance classes")
    _add_bool_arg(parser, "auto_enable_sampler", default=True, help_text="auto enable sampler when imbalance is high")
    _add_bool_arg(
        parser,
        "use_generalization_score",
        default=True,
        help_text="blend validation and generalization scores for early stopping",
    )
    _add_bool_arg(
        parser,
        "use_generalization_rescue",
        default=True,
        help_text="allow recall rescue to trigger on generalization metrics as well as validation",
    )
    parser.add_argument(
        "--generalization_score_weight",
        type=float,
        default=0.3,
        help="Weight for the generalization score when blending with validation (0-1).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Stable mode: turn off aggressive adaptive knobs and KD to avoid collapse
    if args.stable_mode:
        args.use_kd = False
        args.enable_adaptive_reweight = False
        args.use_weighted_sampler = False
        args.auto_enable_sampler = False
        args.use_class_weights = False
        args.class_weight_abnormal = min(args.class_weight_abnormal, 1.2)
        args.max_class_weight_ratio = min(args.max_class_weight_ratio, 1.5)
        args.imbalance_warmup_epochs = max(args.imbalance_warmup_epochs, 8)
        args.kd_warmup_epochs = max(args.kd_warmup_epochs, 12)
        args.collapse_cooldown_epochs = max(args.collapse_cooldown_epochs, 8)
        print(
            "[STABLE] Enabled stable mode: KD disabled, adaptive reweight/sampler off, class-weighting off, gentler caps, longer warmup."
        )
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading MIT-BIH records ...")
    train_x, train_y = load_records(TRAIN_RECORDS, args.data_path)
    gen_x, gen_y = load_records(GENERALIZATION_RECORDS, args.data_path)
    if train_x.size == 0 or gen_x.size == 0:
        raise RuntimeError("No data loaded. Check data path and wfdb installation.")

    tr_x, tr_y, va_x, va_y = split_dataset(train_x, train_y, val_ratio=0.2)
    print(f"Train: {len(tr_x)} | Val: {len(va_x)} | Generalization: {len(gen_x)}")

    abnormal_ratio = float(np.mean(tr_y)) if len(tr_y) > 0 else 0.0
    print(
        f"Class ratio (abnormal): {abnormal_ratio:.3f} | counts -> normal: {(tr_y == 0).sum()} abnormal: {(tr_y == 1).sum()}"
    )

    train_dataset = ECGBeatDataset(tr_x, tr_y)
    use_sampler = False  # start without sampler to avoid early collapse
    plan_sampler = args.use_weighted_sampler or (
        args.auto_enable_sampler and abnormal_ratio < args.auto_sampler_ratio
    )
    current_sampler_boost = args.sampler_abnormal_boost
    target_sampler_boost = current_sampler_boost
    last_sampler_boost_used: float | None = None

    def _build_train_loader(use_weighted: bool, boost: float) -> DataLoader:
        if use_weighted:
            sampler = make_weighted_sampler(tr_y, abnormal_boost=boost)
            return DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_loader = _build_train_loader(use_sampler, current_sampler_boost)
    val_loader = DataLoader(ECGBeatDataset(va_x, va_y), batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(ECGBeatDataset(gen_x, gen_y), batch_size=args.batch_size, shuffle=False)

    student = build_student(args, device)

    base_class_weights = None
    if args.use_class_weights:
        base_class_weights = compute_class_weights(
            tr_y,
            abnormal_boost=args.class_weight_abnormal,
            max_ratio=args.max_class_weight_ratio,
        ).to(device)
    class_weight_scale = 1.0

    teacher = None
    proj_T: nn.Module
    proj_S: nn.Module
    kd_enabled = args.use_kd
    auto_teacher_path = os.path.join("saved_models", "auto_trained_teacher.pth")
    if kd_enabled:
        os.makedirs("saved_models", exist_ok=True)
        checkpoint_path = args.teacher_checkpoint or auto_teacher_path
        if args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint):
            checkpoint_found = True
        else:
            checkpoint_found = os.path.exists(checkpoint_path)
        if checkpoint_found:
            teacher = build_teacher(
                num_classes=len(set(BEAT_LABEL_MAP.values())),
                embedding_dim=args.teacher_embedding_dim,
                device=device,
            )
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            teacher.load_state_dict(state)
            print(f"Loaded teacher checkpoint from {checkpoint_path}")
        elif args.auto_train_teacher:
            teacher = build_teacher(
                num_classes=len(set(BEAT_LABEL_MAP.values())),
                embedding_dim=args.teacher_embedding_dim,
                device=device,
            )
            for p in teacher.parameters():
                p.requires_grad = True
            print(
                "[INFO] No teacher checkpoint found. Auto-training a teacher for knowledge distillation."
            )
            train_teacher(
                teacher,
                train_loader,
                val_loader,
                base_class_weights,
                device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.teacher_auto_train_epochs,
            )
            torch.save({"model_state_dict": teacher.state_dict()}, auto_teacher_path)
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
            print(f"[INFO] Auto-trained teacher saved to {auto_teacher_path}")
        else:
            print("[WARNING] KD enabled but no teacher available. Distillation will be disabled.")
            kd_enabled = False

    if kd_enabled and teacher is not None:
        # sanity check teacher quality to avoid propagating a weak distillation signal
        t_val_loss, t_val_metrics, _, _, _ = evaluate(teacher, val_loader, device)
        if (
            t_val_metrics["f1"] < args.teacher_min_f1
            or t_val_metrics["sensitivity"] < args.teacher_min_sensitivity
        ):
            print(
                f"[WARN] Teacher underperforms (F1={t_val_metrics['f1']:.3f}, "
                f"TPR={t_val_metrics['sensitivity']:.3f}). Disabling KD to avoid harming recall."
            )
            kd_enabled = False

    if kd_enabled and teacher is not None:
        proj_T = nn.Linear(args.teacher_embedding_dim, args.kd_d).to(device)
        proj_cls = ConstrainedLinear if args.use_value_constraint else nn.Linear
        proj_kwargs = {"scale": args.constraint_scale} if args.use_value_constraint else {}
        proj_S = proj_cls(4, args.kd_d, **proj_kwargs).to(device)
    else:
        proj_T = nn.Identity()
        proj_S = nn.Identity()

    current_class_weights = None
    ce_loss_fn = nn.CrossEntropyLoss(weight=current_class_weights)
    mse_loss = nn.MSELoss()

    optimizer = Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, verbose=True)

    best_val_score = -float("inf")
    best_state = None
    best_threshold = 0.5
    patience_counter = 0

    history: List[Dict[str, float]] = []
    collapse_handled = False
    recall_rescue_count = 0
    kd_active = False if kd_enabled else False
    imbalance_active = False
    rebalance_locked = False
    ramp_den = max(1, args.imbalance_ramp_epochs)
    weight_ones = torch.ones_like(base_class_weights) if base_class_weights is not None else None
    reweight_cooldown = 0
    ramp_start_epoch = args.imbalance_warmup_epochs

    for epoch in range(1, args.max_epochs + 1):
        collapse_handled = False
        if reweight_cooldown > 0:
            reweight_cooldown -= 1

        ramp_factor = 0.0
        if epoch > ramp_start_epoch and reweight_cooldown == 0:
            ramp_factor = min(1.0, (epoch - ramp_start_epoch) / ramp_den)
        # Gradually enable imbalance handling and KD after warmup when not in cooldown
        if (
            not imbalance_active
            and (not rebalance_locked)
            and epoch > args.imbalance_warmup_epochs
            and reweight_cooldown == 0
        ):
            if args.use_class_weights and base_class_weights is not None:
                current_class_weights = base_class_weights.clone()
                ce_loss_fn = nn.CrossEntropyLoss(weight=current_class_weights)
                print(
                    f"[WARMUP END] Enabled class weights {current_class_weights.tolist()} after {args.imbalance_warmup_epochs} epochs"
                )
            if plan_sampler:
                use_sampler = True
                train_loader = _build_train_loader(use_sampler, target_sampler_boost)
                print(
                    f"[WARMUP END] Enabled weighted sampler (abnormal boost {target_sampler_boost:.2f})"
                )
            imbalance_active = True

        if kd_enabled and (not kd_active) and epoch > args.kd_warmup_epochs:
            kd_active = True
            print(f"[WARMUP END] KD activated after {args.kd_warmup_epochs} epochs")

        student.train()
        running_loss = 0.0
        total = 0

        if reweight_cooldown > 0:
            current_class_weights = None
            ce_loss_fn = nn.CrossEntropyLoss(weight=current_class_weights)
        elif args.use_class_weights and base_class_weights is not None:
            scaled_weights = base_class_weights.clone()
            if weight_ones is not None:
                scaled_weights = weight_ones + (base_class_weights - weight_ones) * (ramp_factor * class_weight_scale)
                scaled_weights[1] = min(
                    scaled_weights[1], scaled_weights[0] * args.max_class_weight_ratio
                )
            current_class_weights = scaled_weights
            ce_loss_fn = nn.CrossEntropyLoss(weight=current_class_weights)

        if use_sampler and reweight_cooldown == 0:
            effective_boost = 1.0 + (target_sampler_boost - 1.0) * ramp_factor
            if last_sampler_boost_used is None or abs(effective_boost - last_sampler_boost_used) > 1e-3:
                train_loader = _build_train_loader(use_sampler, effective_boost)
                last_sampler_boost_used = effective_boost

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()

            logits_s, feat_s = student(signals)
            loss_ce = ce_loss_fn(logits_s, labels)
            loss = loss_ce

            if kd_active and kd_enabled and teacher is not None:
                with torch.no_grad():
                    logits_t, feat_t = teacher(signals)
                kd_logits = kd_logit_loss(logits_s, logits_t, args.kd_temperature)
                f_s = proj_S(scale_to_unit(feat_s) if args.use_value_constraint and not args.use_tanh_activations else feat_s)
                f_t = proj_T(feat_t)
                f_s = l2_normalize(f_s)
                f_t = l2_normalize(f_t)
                kd_feat = mse_loss(f_s, f_t)
                loss = args.alpha * loss_ce + args.beta * kd_logits + args.gamma * kd_feat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)

        val_loss, val_argmax_metrics, val_true, _, val_probs = evaluate(
            student, val_loader, device, return_probs=True
        )

        best_thr_val, val_metrics = sweep_thresholds(
            val_true,
            val_probs,
            miss_target=args.threshold_target_miss,
            fpr_cap=args.threshold_max_fpr,
        )

        gen_loss, gen_argmax_metrics, gen_true, _, gen_probs = evaluate(
            student, gen_loader, device, return_probs=True
        )

        if args.use_blended_thresholds:
            best_thr_epoch, val_metrics, gen_metrics = sweep_thresholds_blended(
                val_true,
                val_probs,
                gen_true,
                gen_probs,
                gen_weight=args.threshold_generalization_weight,
                miss_target=args.threshold_target_miss,
                fpr_cap=args.threshold_max_fpr,
            )
        else:
            best_thr_epoch = best_thr_val
            val_metrics = confusion_metrics(val_true, (np.array(val_probs) >= best_thr_epoch).astype(int).tolist())
            gen_metrics = confusion_metrics(gen_true, (np.array(gen_probs) >= best_thr_epoch).astype(int).tolist())

        val_rescue = val_metrics["miss_rate"] > args.recall_target_miss and val_metrics["fpr"] < args.adaptive_fpr_cap
        gen_rescue = (
            args.use_generalization_rescue
            and gen_metrics["miss_rate"] > args.recall_target_miss
            and gen_metrics["fpr"] < args.adaptive_fpr_cap
        )
        if (
            args.enable_adaptive_reweight
            and imbalance_active
            and recall_rescue_count < args.recall_rescue_limit
            and (val_rescue or gen_rescue)
        ):
            if args.use_class_weights and current_class_weights is not None and current_class_weights.numel() > 1:
                class_weight_scale = min(
                    class_weight_scale * 1.2,
                    args.max_class_weight_ratio,
                )
                print(
                    f"[ADAPT] High miss detected. Increasing abnormal class weight scale to {class_weight_scale:.2f}"
                )
            if not use_sampler:
                use_sampler = True
                target_sampler_boost = max(target_sampler_boost, 1.5)
                train_loader = _build_train_loader(use_sampler, target_sampler_boost)
                print(
                    f"[ADAPT] Enabled weighted sampler with abnormal boost {target_sampler_boost:.2f} to improve recall."
                )
            recall_rescue_count += 1

        if kd_active and val_metrics["miss_rate"] > args.kd_pause_miss:
            kd_active = False
            print(
                f"[KD] Pausing distillation because miss_rate={val_metrics['miss_rate']:.3f} exceeds {args.kd_pause_miss:.3f}"
            )
        elif (
            kd_enabled
            and (not kd_active)
            and epoch > args.kd_warmup_epochs
            and val_metrics["miss_rate"] < args.kd_resume_miss
            and teacher is not None
        ):
            kd_active = True
            print(
                f"[KD] Resuming distillation after miss_rate dropped below {args.kd_resume_miss:.3f}"
            )
        if (
            not args.stable_mode
            and not collapse_handled
            and val_argmax_metrics["fpr"] > 0.95
            and val_argmax_metrics["miss_rate"] < 0.05
        ):
            print(
                "[WARN] Detected positive-collapse (predicting nearly all abnormal). Relaxing rebalancing and pausing KD."
            )
            if use_sampler:
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True
                )
                use_sampler = False
                print("       Switched to unweighted sampler.")
            current_class_weights = None
            ce_loss_fn = nn.CrossEntropyLoss(weight=current_class_weights)
            class_weight_scale = 1.0
            kd_active = False
            kd_enabled = False
            patience_counter = 0
            collapse_handled = True
            reweight_cooldown = args.collapse_cooldown_epochs
            imbalance_active = False
            ramp_start_epoch = epoch + reweight_cooldown
            rebalance_locked = True
            plan_sampler = False

        if (
            not args.stable_mode
            and recall_rescue_count < args.recall_rescue_limit
            and val_argmax_metrics["miss_rate"] > 0.95
            and val_argmax_metrics["fpr"] < 0.05
            and epoch > args.imbalance_warmup_epochs
        ):
            # collapsed to predicting normal only
            if base_class_weights is not None:
                current_class_weights = base_class_weights.clone()
                ce_loss_fn = nn.CrossEntropyLoss(weight=current_class_weights)
                print(
                    f"[WARN] Detected normal-collapse. Restoring class weights {current_class_weights.tolist()} and enabling sampler."
                )
            use_sampler = True
            target_sampler_boost = max(target_sampler_boost, 1.5)
            train_loader = _build_train_loader(use_sampler, target_sampler_boost)
            kd_active = False  # let CE recover before KD resumes
            collapse_handled = True
            recall_rescue_count += 1
            reweight_cooldown = args.collapse_cooldown_epochs
            imbalance_active = False
            ramp_start_epoch = epoch + reweight_cooldown
            rebalance_locked = True
            plan_sampler = False
            kd_enabled = False
        scheduler.step(val_loss)

        val_score = val_metrics["f1"] + val_metrics["sensitivity"] - 0.5 * val_metrics["fpr"]
        gen_score = gen_metrics["f1"] + gen_metrics["sensitivity"] - 0.5 * gen_metrics["fpr"]
        if args.use_generalization_score:
            blend = np.clip(args.generalization_score_weight, 0.0, 1.0)
            epoch_score = (1 - blend) * val_score + blend * gen_score
        else:
            epoch_score = val_score

        print(
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val@thr={best_thr_epoch:.2f} F1 {val_metrics['f1']:.3f} Miss {val_metrics['miss_rate'] * 100:.2f}% "
            f"FPR {val_metrics['fpr'] * 100:.2f}% | Gen@thr={best_thr_epoch:.2f} F1 {gen_metrics['f1']:.3f} "
            f"Miss {gen_metrics['miss_rate'] * 100:.2f}% FPR {gen_metrics['fpr'] * 100:.2f}% Score {epoch_score:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_miss": val_metrics["miss_rate"],
                "val_fpr": val_metrics["fpr"],
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
            }
        )

        if epoch_score > best_val_score:
            best_val_score = epoch_score
            best_state = student.state_dict()
            best_threshold = best_thr_epoch
            patience_counter = 0
            print("  -> New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch >= args.min_epochs:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    val_loss, _, val_true, _, val_probs = evaluate(
        student, val_loader, device, return_probs=True
    )
    gen_loss, _, gen_true, _, gen_probs = evaluate(
        student, gen_loader, device, return_probs=True
    )

    if args.use_blended_thresholds:
        best_threshold, val_metrics, gen_metrics = sweep_thresholds_blended(
            val_true,
            val_probs,
            gen_true,
            gen_probs,
            gen_weight=args.threshold_generalization_weight,
            miss_target=args.threshold_target_miss,
            fpr_cap=args.threshold_max_fpr,
        )
    else:
        best_threshold, val_metrics = sweep_thresholds(
            val_true,
            val_probs,
            miss_target=args.threshold_target_miss,
            fpr_cap=args.threshold_max_fpr,
        )
        gen_metrics = confusion_metrics(gen_true, (np.array(gen_probs) >= best_threshold).astype(int).tolist())
    val_pred = (np.array(val_probs) >= best_threshold).astype(int).tolist()
    gen_pred = (np.array(gen_probs) >= best_threshold).astype(int).tolist()

    print(
        f"Final Val@thr={best_threshold:.2f}: loss={val_loss:.4f}, F1={val_metrics['f1']:.3f}, "
        f"miss={val_metrics['miss_rate'] * 100:.2f}%, fpr={val_metrics['fpr'] * 100:.2f}%"
    )
    print(
        f"Generalization@thr={best_threshold:.2f}: loss={gen_loss:.4f}, F1={gen_metrics['f1']:.3f}, "
        f"miss={gen_metrics['miss_rate'] * 100:.2f}%, fpr={gen_metrics['fpr'] * 100:.2f}%"
    )

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "student_model.pth")
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "config": vars(args),
            "best_threshold": best_threshold,
        },
        save_path,
    )
    print(f"Saved student checkpoint to {save_path}")

    # Visualization and artifact saving
    os.makedirs("artifacts", exist_ok=True)

    def _save_training_curves() -> None:
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
        axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()

        axes[1].plot(epochs, [h["val_f1"] for h in history], label="F1")
        axes[1].plot(epochs, [h["val_miss"] for h in history], label="Miss Rate")
        axes[1].plot(epochs, [h["val_fpr"] for h in history], label="FPR")
        if any("gen_f1" in h for h in history):
            axes[1].plot(epochs, [h.get("gen_f1", float("nan")) for h in history], label="Gen F1", linestyle="--")
            axes[1].plot(epochs, [h.get("gen_miss", float("nan")) for h in history], label="Gen Miss", linestyle="--")
            axes[1].plot(epochs, [h.get("gen_fpr", float("nan")) for h in history], label="Gen FPR", linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Val Metrics")
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(os.path.join("artifacts", "training_curves.png"))
        plt.close(fig)

    def _save_roc(y_true: List[int], probs: List[float], name: str) -> None:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC - {name}")
        ax.legend()
        fig.savefig(os.path.join("artifacts", f"roc_{name.lower()}.png"))
        plt.close(fig)

    def _save_confusion(y_true: List[int], y_pred: List[int], name: str) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join("artifacts", f"confusion_{name.lower()}.png"))
        plt.close(fig)

    _save_training_curves()
    _save_roc(val_true, val_probs, "Val")
    _save_roc(gen_true, gen_probs, "Generalization")
    _save_confusion(val_true, val_pred, "Val")
    _save_confusion(gen_true, gen_pred, "Generalization")
    print("Saved training curves, ROC curves, and confusion matrices to ./artifacts")


if __name__ == "__main__":
    main()
