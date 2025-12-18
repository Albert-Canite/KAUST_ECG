"""Training script for segment-aware ECG classification with plain cross-entropy."""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from utils import (
    FocalLoss,
    compute_class_weights,
    compute_multiclass_metrics,
    make_weighted_sampler,
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

NUM_CLASSES = len(set(BEAT_LABEL_MAP.values()))
CLASS_NAMES = ["N", "S", "V", "O"]


def build_student(args: argparse.Namespace, device: torch.device) -> nn.Module:
    student = SegmentAwareStudent(
        num_classes=NUM_CLASSES,
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
    num_classes: int,
) -> Tuple[float, Dict[str, float], List[int], List[int], torch.Tensor]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    preds: List[int] = []
    trues: List[int] = []
    probs_all: List[torch.Tensor] = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            prob_all = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            probs_all.append(prob_all.detach().cpu())
    avg_loss = total_loss / max(total, 1)
    metrics = compute_multiclass_metrics(trues, preds, num_classes)  # type: ignore[arg-type]

    stacked_probs = torch.cat(probs_all, dim=0) if probs_all else torch.empty(0, num_classes)
    return avg_loss, metrics, trues, preds, stacked_probs


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Backward-compatible boolean flags with --name / --no-name.

    Guard against double registration (seen in some downstream wrappers) by
    skipping if the positive flag already exists.
    """

    opt = f"--{name}"
    if any(opt in action.option_strings for action in parser._actions):
        parser.set_defaults(**{name: default})
        return

    parser.add_argument(opt, dest=name, action="store_true", help=f"Enable {help_text}")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIT-BIH ECG training with cross-entropy baseline", conflict_handler="resolve"
    )
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
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument(
        "--class_weight_max_ratio",
        type=float,
        default=20.0,
        help="cap for CE class weights; set to 0 or omit to disable clamping for extreme imbalance",
    )
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=0.5,
        help="inverse-frequency exponent for CE weights (0.5=sqrt, 1.0=full)",
    )
    parser.add_argument(
        "--weight_warmup_epochs",
        type=int,
        default=0,
        help="epochs to linearly warm class weights from uniform to target inverse-frequency weights (0 disables)",
    )
    _add_bool_arg(
        parser,
        "use_weighted_sampler",
        default=False,
        help_text="enable weighted sampler (sqrt balancing) for long-tail classes; disable to rely on CE class weights",
    )
    parser.add_argument(
        "--sampler_power",
        type=float,
        default=0.5,
        help="inverse-frequency exponent for sampler (0.5=sqrt, 1.0=full balance)",
    )
    parser.add_argument(
        "--sampler_max_ratio",
        type=float,
        default=6.0,
        help="cap on class sampling weights to avoid overshooting ultra-rare beats",
    )
    parser.add_argument(
        "--curriculum_epochs",
        type=int,
        default=6,
        help=(
            "use a mild balanced sampler for the first N epochs to guarantee early S/V exposure; "
            "set to 0 to disable the curriculum stage"
        ),
    )
    parser.add_argument(
        "--curriculum_sampler_power",
        type=float,
        default=0.5,
        help="inverse-frequency exponent for the curriculum sampler (0.5=sqrt balance)",
    )
    parser.add_argument(
        "--curriculum_sampler_max_ratio",
        type=float,
        default=4.0,
        help="cap for curriculum sampler weights to prevent runaway oversampling",
    )
    parser.add_argument(
        "--target_sampler_mix",
        type=str,
        default="",
        help=(
            "optional comma-separated target batch mix for (N,S,V,O); empty string disables and uses inverse-frequency sampler"
        ),
    )
    _add_bool_arg(
        parser,
        "stack_sampler_with_ce",
        default=False,
        help_text=(
            "stack inverse-frequency CE weights with the weighted sampler (disabled by default to avoid over-correction)"
        ),
    )
    parser.add_argument(
        "--loss_type",
        choices=["ce", "focal"],
        default="ce",
        help="loss to train the student (focal improves hard S/V recall)",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="gamma for focal loss")
    parser.add_argument("--seed", type=int, default=42)
    _add_bool_arg(parser, "use_value_constraint", default=True, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    def _print_class_stats(name: str, labels: np.ndarray) -> None:
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        total = counts.sum()
        print(f"{name} class counts (N,S,V,O): {counts.tolist()} | total={int(total)}")

    _print_class_stats("Train", tr_y)
    _print_class_stats("Val", va_y)
    _print_class_stats("Gen", gen_y)

    class_counts = np.bincount(tr_y, minlength=NUM_CLASSES)
    total_counts = class_counts.sum()
    abnormal_ratio = 1.0 - (class_counts[0] / total_counts) if total_counts > 0 else 0.0
    print(
        "Class distribution (N,S,V,O): "
        f"{class_counts.tolist()} | non-normal fraction={abnormal_ratio:.3f}"
    )
    missing_train = [CLASS_NAMES[i] for i, cnt in enumerate(class_counts) if cnt == 0]
    if missing_train:
        raise RuntimeError(
            "Training split is missing classes: "
            f"{missing_train}. Retry with different records or disable stratified split to avoid zero-shot classes."
        )

    train_dataset = ECGBeatDataset(tr_x, tr_y)

    sampler = None
    sampler_weights = None
    sampler_mix = None
    target_mix = None
    balanced_sampler = None
    balanced_sampler_weights = None
    balanced_sampler_mix = None
    if args.target_sampler_mix.strip():
        target_raw = np.array([float(x) for x in args.target_sampler_mix.split(",")], dtype=float)
        if target_raw.sum() <= 0:
            raise ValueError("target_sampler_mix must sum to a positive value")
        target_mix = (target_raw / target_raw.sum()).tolist()
        if len(target_mix) != NUM_CLASSES:
            raise ValueError(
                f"target_sampler_mix must provide {NUM_CLASSES} values for N/S/V/O, got {len(target_mix)}"
            )
    if args.use_weighted_sampler:
        sampler, sampler_weights, sampler_mix = make_weighted_sampler(
            tr_y,
            power=args.sampler_power,
            max_ratio=args.sampler_max_ratio,
            target_mix=target_mix,
            num_classes=NUM_CLASSES,
        )
        print(
            "Using weighted sampler for 4-class to surface rare S/V beats. "
            f"power={args.sampler_power:.2f}, max_ratio={args.sampler_max_ratio:.1f}"
        )
        if target_mix is not None:
            print(f"Target sampler mix (N,S,V,O): {[round(m,3) for m in target_mix]} (normalized)")
        print(
            "Per-class sampler weights (post-clamp): "
            f"{[round(sampler_weights.get(cid, 0.0), 3) for cid in range(NUM_CLASSES)]}"
        )
        if sampler_mix:
            mix_pct = [round(100 * sampler_mix.get(cid, 0.0), 1) for cid in range(NUM_CLASSES)]
            print(f"Expected batch mix from sampler (N,S,V,O): {mix_pct} %")

    # Mild curriculum sampler to expose S/V early without forcing balance for the whole run.
    # Only apply when the primary sampler is off to avoid stacking two resampling strategies.
    if args.curriculum_epochs > 0 and sampler is None:
        balanced_sampler, balanced_sampler_weights, balanced_sampler_mix = make_weighted_sampler(
            tr_y,
            power=args.curriculum_sampler_power,
            max_ratio=args.curriculum_sampler_max_ratio,
            num_classes=NUM_CLASSES,
        )
        print(
            "Curriculum sampler enabled for initial epochs to ensure rare-class exposure: "
            f"epochs=1-{args.curriculum_epochs}, power={args.curriculum_sampler_power:.2f}, "
            f"max_ratio={args.curriculum_sampler_max_ratio:.1f}"
        )
        print(
            "Curriculum sampler weights (post-clamp): "
            f"{[round(balanced_sampler_weights.get(cid, 0.0), 3) for cid in range(NUM_CLASSES)]}"
        )
        if balanced_sampler_mix:
            mix_pct = [round(100 * balanced_sampler_mix.get(cid, 0.0), 1) for cid in range(NUM_CLASSES)]
            print(f"Expected batch mix during curriculum (N,S,V,O): {mix_pct} %")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    curriculum_loader = None
    if balanced_sampler is not None:
        curriculum_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=balanced_sampler,
        )
    val_loader = DataLoader(ECGBeatDataset(va_x, va_y), batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(ECGBeatDataset(gen_x, gen_y), batch_size=args.batch_size, shuffle=False)

    # Keep loss reflecting class prior with inverse-frequency weights (no binary boosts).
    class_weights_np = compute_class_weights(
        tr_y,
        max_ratio=args.class_weight_max_ratio,
        num_classes=NUM_CLASSES,
        power=args.class_weight_power,
    )
    raw_weights = []
    max_count = max(class_counts) if len(class_counts) else 0
    for count in class_counts:
        base = (max_count / max(count, 1)) ** args.class_weight_power
        raw_weights.append(base)
    clamping_on = args.class_weight_max_ratio is not None and args.class_weight_max_ratio > 0
    min_w = 1.0
    max_w = args.class_weight_max_ratio if clamping_on else float("inf")
    print(
        f"Class weights computed vs. majority count (power={args.class_weight_power:.2f}): "
        f"raw={np.round(raw_weights, 4)}"
    )
    print(
        f"Clamped to [1, {args.class_weight_max_ratio}] to avoid downweighting N: "
        f"final weights={np.round(class_weights_np.cpu().numpy(), 4)} (min={min_w:.1f}, max={max_w})"
    )

    class_weights = class_weights_np.to(device)
    # Keep some class weighting even when a sampler is on; a soft exponent avoids over-correction
    # while preventing the sampler from drifting to all-N/O solutions. When the curriculum sampler
    # is used, we stay conservative to avoid repeating earlier over-balancing failures.
    if sampler is not None and not args.stack_sampler_with_ce:
        base_weights = torch.pow(class_weights, 0.5)
        print(
            "Sampler active -> applying mild CE reweighting (sqrt of inverse-freq) to reinforce S/V. "
            f"Effective CE weights: {np.round(base_weights.cpu().numpy(), 4)}"
        )
    else:
        base_weights = class_weights.clone()
        if sampler is not None:
            print(
                "Sampler active **and** CE stacking enabled -> combining sampler upsampling with "
                f"full inverse-freq CE weights: {np.round(class_weights.cpu().numpy(), 4)}"
            )

    curriculum_weights = torch.pow(class_weights, 0.5)
    if curriculum_loader is not None:
        print(
            "Curriculum stage CE weights (sqrt inverse-freq) to pair with mild sampler: "
            f"{np.round(curriculum_weights.cpu().numpy(), 4)}"
        )

    os.makedirs("artifacts", exist_ok=True)
    log_path = os.path.join(
        "artifacts",
        f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    student = build_student(args, device)

    optimizer = Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, verbose=True)

    def _write_log(entry: Dict[str, object]) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    data_min = float(np.min(train_dataset.data)) if len(train_dataset) > 0 else 0.0
    data_max = float(np.max(train_dataset.data)) if len(train_dataset) > 0 else 0.0

    _write_log(
        {
            "event": "start",
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.detach().cpu().tolist(),
            "ce_weights": base_weights.detach().cpu().tolist(),
            "sampler_weights": sampler_weights if sampler_weights is not None else None,
            "sampler_mix": sampler_mix if sampler_mix is not None else None,
            "data_range": [data_min, data_max],
        }
    )

    print(f"Preprocessed input range: [{data_min:.3f}, {data_max:.3f}] (expected within [-1, 1])")

    best_checkpoint_score = -float("inf")
    best_state = None
    patience_counter = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        running_loss = 0.0
        total = 0

        use_curriculum = curriculum_loader is not None and epoch <= args.curriculum_epochs
        epoch_loader = curriculum_loader if use_curriculum else train_loader
        epoch_base_weights = curriculum_weights if use_curriculum else base_weights

        if args.weight_warmup_epochs > 0:
            warm_frac = min(1.0, max(0.0, (epoch - 1) / float(args.weight_warmup_epochs)))
            epoch_weights = torch.ones_like(epoch_base_weights) * (1 - warm_frac) + epoch_base_weights * warm_frac
            warm_frac_display = warm_frac
        else:
            epoch_weights = epoch_base_weights.clone()
            warm_frac_display = 1.0
        if args.loss_type == "focal":
            alpha = epoch_weights / max(epoch_weights.sum(), 1e-8)
            loss_fn = FocalLoss(alpha=alpha.to(device), gamma=args.focal_gamma)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=epoch_weights)
        if epoch == 1 or (args.weight_warmup_epochs > 0 and warm_frac < 1.0 and epoch % 5 == 0):
            print(
                f"Epoch {epoch}: weight warmup alpha={warm_frac_display:.2f}, weights={epoch_weights.detach().cpu().numpy()}"
            )
            print(f"  -> Using {args.loss_type.upper()} loss (gamma={args.focal_gamma:.2f} when focal)")

        if use_curriculum and epoch == 1:
            print(
                "Curriculum phase active: using mild sampler and sqrt weights for early rare-class exposure"
            )
        if use_curriculum and epoch == args.curriculum_epochs:
            print("Last curriculum epoch before switching to natural sampling next epoch")

        _write_log(
            {
                "event": "epoch_start",
                "epoch": epoch,
                "use_curriculum": use_curriculum,
                "warmup_alpha": warm_frac_display,
                "ce_weights": epoch_weights.detach().cpu().tolist(),
                "stack_sampler_with_ce": args.stack_sampler_with_ce,
            }
        )

        for signals, labels in epoch_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, _ = student(signals)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)

        val_loss, val_metrics_mc, val_true, val_pred_mc, _ = evaluate(
            student, val_loader, device, NUM_CLASSES
        )
        gen_loss, gen_metrics_mc, gen_true, gen_pred_mc, _ = evaluate(
            student, gen_loader, device, NUM_CLASSES
        )

        val_abnormal_f1 = val_metrics_mc.get("abnormal_macro_f1", val_metrics_mc["macro_f1"])
        gen_abnormal_f1 = gen_metrics_mc.get("abnormal_macro_f1", gen_metrics_mc["macro_f1"])
        val_rare_f1 = val_metrics_mc.get("rare_macro_f1", val_abnormal_f1)
        gen_rare_f1 = gen_metrics_mc.get("rare_macro_f1", gen_abnormal_f1)

        # Penalize any class collapse by tracking the minimum per-class F1.
        val_min_f1 = min((m.get("f1", 0.0) for m in val_metrics_mc.get("per_class", {}).values()), default=0.0)
        gen_min_f1 = min((m.get("f1", 0.0) for m in gen_metrics_mc.get("per_class", {}).values()), default=0.0)

        # Encourage balanced learning across all classes; abnormal beats still matter,
        # but ignoring any class (including O) lowers the checkpoint score.
        composite_score = (
            0.4 * val_metrics_mc["macro_f1"]
            + 0.3 * val_abnormal_f1
            + 0.3 * val_min_f1
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val MacroF1 {val_metrics_mc['macro_f1']:.3f} (abn {val_abnormal_f1:.3f} rare {val_rare_f1:.3f} min {val_min_f1:.3f}) Acc {val_metrics_mc['accuracy']:.3f} | "
            f"Gen MacroF1 {gen_metrics_mc['macro_f1']:.3f} (abn {gen_abnormal_f1:.3f} rare {gen_rare_f1:.3f} min {gen_min_f1:.3f}) Acc {gen_metrics_mc['accuracy']:.3f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_metrics_mc["macro_f1"],
                "val_acc": val_metrics_mc["accuracy"],
                "gen_macro_f1": gen_metrics_mc["macro_f1"],
                "gen_acc": gen_metrics_mc["accuracy"],
                "val_abnormal_macro_f1": val_abnormal_f1,
                "val_rare_macro_f1": val_rare_f1,
                "val_min_f1": val_min_f1,
                "gen_abnormal_macro_f1": gen_abnormal_f1,
                "gen_rare_macro_f1": gen_rare_f1,
                "gen_min_f1": gen_min_f1,
                "checkpoint_score": composite_score,
            }
        )

        _write_log(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_metrics_mc["macro_f1"],
                "val_acc": val_metrics_mc["accuracy"],
                "val_abnormal_macro_f1": val_abnormal_f1,
                "gen_macro_f1": gen_metrics_mc["macro_f1"],
                "gen_abnormal_macro_f1": gen_abnormal_f1,
                "gen_rare_macro_f1": gen_rare_f1,
                "val_min_f1": val_min_f1,
                "gen_min_f1": gen_min_f1,
                "gen_acc": gen_metrics_mc["accuracy"],
                "checkpoint_score": composite_score,
            }
        )

        # 4-class monitoring: emphasize abnormal beats by blending macro F1
        # with abnormal-only macro F1. This prevents degenerate "all N" models
        # from winning early stopping when rare classes are ignored.
        if composite_score > best_checkpoint_score:
            best_checkpoint_score = composite_score
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
            print(
                "  -> New best model saved (by blended Val MacroF1 + abnormal MacroF1)."
            )
            _write_log(
                {
                    "event": "best",
                    "epoch": epoch,
                "val_macro_f1": val_metrics_mc["macro_f1"],
                "val_acc": val_metrics_mc["accuracy"],
                "val_abnormal_macro_f1": val_abnormal_f1,
                "val_rare_macro_f1": val_rare_f1,
                "gen_macro_f1": gen_metrics_mc["macro_f1"],
                "gen_abnormal_macro_f1": gen_abnormal_f1,
                "gen_rare_macro_f1": gen_rare_f1,
                "gen_acc": gen_metrics_mc["accuracy"],
                "checkpoint_score": composite_score,
            }
        )
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch >= args.min_epochs:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    val_loss, val_metrics_mc, val_true, val_pred_mc, _ = evaluate(
        student, val_loader, device, NUM_CLASSES
    )
    gen_loss, gen_metrics_mc, gen_true, gen_pred_mc, _ = evaluate(
        student, gen_loader, device, NUM_CLASSES
    )

    print(
        f"Validation multi-class: Acc={val_metrics_mc['accuracy']:.3f}, MacroF1={val_metrics_mc['macro_f1']:.3f}"
    )
    print(
        f"Generalization multi-class: Acc={gen_metrics_mc['accuracy']:.3f}, MacroF1={gen_metrics_mc['macro_f1']:.3f}"
    )

    val_cm = confusion_matrix(val_true, val_pred_mc, labels=list(range(NUM_CLASSES)))
    gen_cm = confusion_matrix(gen_true, gen_pred_mc, labels=list(range(NUM_CLASSES)))
    print(f"Validation 4-class confusion matrix (rows=true, cols=pred):\n{val_cm}")
    for cid in range(NUM_CLASSES):
        mc = val_metrics_mc.get("per_class", {}).get(cid, {})
        print(
            f"Val class {CLASS_NAMES[cid]}: precision={mc.get('precision', 0):.3f} "
            f"recall={mc.get('recall', 0):.3f} f1={mc.get('f1', 0):.3f}"
        )
    print(f"Generalization 4-class confusion matrix (rows=true, cols=pred):\n{gen_cm}")
    for cid in range(NUM_CLASSES):
        mc = gen_metrics_mc.get("per_class", {}).get(cid, {})
        print(
            f"Gen class {CLASS_NAMES[cid]}: precision={mc.get('precision', 0):.3f} "
            f"recall={mc.get('recall', 0):.3f} f1={mc.get('f1', 0):.3f}"
        )

    _write_log(
        {
            "event": "final",
            "val_loss": val_loss,
            "gen_loss": gen_loss,
            "val_macro_f1": val_metrics_mc["macro_f1"],
            "val_acc": val_metrics_mc["accuracy"],
            "gen_macro_f1": gen_metrics_mc["macro_f1"],
            "gen_acc": gen_metrics_mc["accuracy"],
        }
    )

    # Persist labels for offline diagnostics
    np.save(os.path.join("artifacts", "val_labels.npy"), np.array(val_true))
    np.save(os.path.join("artifacts", "gen_labels.npy"), np.array(gen_true))

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "student_model.pth")
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "config": vars(args),
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

        axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="Val MacroF1")
        axes[1].plot(epochs, [h["val_acc"] for h in history], label="Val Acc")
        if any("gen_macro_f1" in h for h in history):
            axes[1].plot(epochs, [h.get("gen_macro_f1", float("nan")) for h in history], label="Gen MacroF1", linestyle="--")
            axes[1].plot(epochs, [h.get("gen_acc", float("nan")) for h in history], label="Gen Acc", linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Val Metrics")
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(os.path.join("artifacts", "training_curves.png"))
        plt.close(fig)

    def _save_confusion(y_true: List[int], y_pred: List[int], name: str, labels: List[int], class_names: List[str]) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {name}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join("artifacts", f"confusion_{name.lower()}.png"))
        plt.close(fig)

    _save_training_curves()
    _save_confusion(val_true, val_pred_mc, "Val_4class", list(range(NUM_CLASSES)), CLASS_NAMES)
    _save_confusion(gen_true, gen_pred_mc, "Generalization_4class", list(range(NUM_CLASSES)), CLASS_NAMES)
    print("Saved training curves and confusion matrices to ./artifacts")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
