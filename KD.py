from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from models.teacher import resnet18_1d
from train import GENERALIZATION_RECORDS, TRAIN_RECORDS, build_student
from utils import compute_class_weights, compute_multiclass_metrics, kd_logit_loss, make_weighted_sampler


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    gen_loader: DataLoader
    val_arrays: Tuple[np.ndarray, np.ndarray]
    gen_arrays: Tuple[np.ndarray, np.ndarray]
    class_weights: torch.Tensor


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Boolean flags with safety against duplicate registration."""

    opt = f"--{name}"
    if any(opt in action.option_strings for action in parser._actions):
        parser.set_defaults(**{name: default})
        return

    parser.add_argument(opt, dest=name, action="store_true", help=f"Enable {help_text}")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplified KD script (multiclass evaluation only)", conflict_handler="resolve"
    )
    parser.add_argument("--data_path", type=str, default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_mlp_layers", type=int, default=3)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--class_weight_max_ratio", type=float, default=5.0)
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=1.0,
        help="inverse-frequency exponent for CE weights (0.5=sqrt, 1.0=full)",
    )
    parser.add_argument(
        "--weight_warmup_epochs",
        type=int,
        default=8,
        help="epochs to linearly warm class weights from uniform to target inverse-frequency weights",
    )
    _add_bool_arg(parser, "use_value_constraint", default=True, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")
    _add_bool_arg(
        parser,
        "use_weighted_sampler",
        default=False,
        help_text="enable weighted sampler (sqrt balancing) for long-tail classes; default off to preserve normal prior",
    )
    parser.add_argument("--sampler_power", type=float, default=0.5)
    parser.add_argument("--student_path", type=str, default=os.path.join("saved_models", "student_model.pth"))
    parser.add_argument("--teacher_path", type=str, default=os.path.join("saved_models", "teacher_model.pth"))
    parser.add_argument("--kd_alpha", type=float, default=0.35, help="weight for KD logit loss vs CE")
    parser.add_argument("--kd_temperature", type=float, default=4.0, help="temperature for KD logit loss")
    parser.add_argument("--teacher_max_epochs", type=int, default=25)
    parser.add_argument("--teacher_patience", type=int, default=6)
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace, device: torch.device) -> DatasetBundle:
    train_x, train_y = load_records(TRAIN_RECORDS, args.data_path)
    gen_x, gen_y = load_records(GENERALIZATION_RECORDS, args.data_path)
    if train_x.size == 0 or gen_x.size == 0:
        raise RuntimeError("No data loaded. Check data path and wfdb installation.")

    tr_x, tr_y, va_x, va_y = split_dataset(train_x, train_y, val_ratio=0.2)

    train_dataset = ECGBeatDataset(tr_x, tr_y)
    sampler = None
    num_classes = len(set(BEAT_LABEL_MAP.values()))
    class_counts = np.bincount(tr_y, minlength=num_classes)
    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(tr_y, power=args.sampler_power)
        print(
            "[KD] Using weighted sampler for 4-class to expose rare S/V beats; CE will stay uniform to avoid double balancing. "
            f"power={args.sampler_power:.2f}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    val_loader = DataLoader(ECGBeatDataset(va_x, va_y), batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(ECGBeatDataset(gen_x, gen_y), batch_size=args.batch_size, shuffle=False)

    class_weights_np = compute_class_weights(
        tr_y,
        max_ratio=args.class_weight_max_ratio,
        num_classes=num_classes,
        power=args.class_weight_power,
    )
    class_weights = class_weights_np.to(device)
    if sampler is not None:
        ce_weights = torch.ones_like(class_weights)
        print(
            "[KD] Sampler active -> using uniform CE weights (sampler handles imbalance). "
            f"Original inverse-freq weights: {np.round(class_weights.cpu().numpy(), 4)}"
        )
    else:
        ce_weights = class_weights
    print(
        "[KD] Train class counts (N,S,V,O): "
        f"{class_counts.tolist()} | weights={np.round(ce_weights.cpu().numpy(), 4)}"
    )

    return DatasetBundle(train_loader, val_loader, gen_loader, (va_x, va_y), (gen_x, gen_y), ce_weights)


def evaluate_with_probs(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    preds: List[int] = []
    trues: List[int] = []
    sample_debug: Dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            if sample_debug is None:
                sample_debug = {
                    "y_true": labels.detach().cpu(),
                    "pred": pred.detach().cpu(),
                    "prob_first_class": torch.softmax(logits, dim=1)[:, 0].detach().cpu(),
                }
    avg_loss = total_loss / max(total, 1)
    metrics = compute_multiclass_metrics(trues, preds, num_classes=len(set(BEAT_LABEL_MAP.values())))

    cm = confusion_matrix(trues, preds, labels=list(range(len(set(BEAT_LABEL_MAP.values())))))
    print(f"[KD] 4-class confusion matrix (rows=true, cols=pred):\n{cm}")
    for cid, cname in enumerate(["N", "S", "V", "O"]):
        mc = metrics.get("per_class", {}).get(cid, {})
        print(
            f"[KD] Class {cname}: precision={mc.get('precision', 0):.3f} "
            f"recall={mc.get('recall', 0):.3f} f1={mc.get('f1', 0):.3f}"
        )
    if sample_debug is not None:
        print(
            "[KD] Sample sanity (first batch, first 10):",
            "y_true=", sample_debug["y_true"][:10].tolist(),
            "pred=", sample_debug["pred"][:10].tolist(),
            "prob_first_class=", sample_debug["prob_first_class"][:10].tolist(),
        )

    return avg_loss, metrics, trues, preds


def load_student(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, argparse.Namespace]:
    if not os.path.exists(args.student_path):
        raise FileNotFoundError(f"Student checkpoint not found at {args.student_path}")
    ckpt = torch.load(args.student_path, map_location=device)
    config = argparse.Namespace(**ckpt.get("config", {})) if "config" in ckpt else args
    config.dropout_rate = getattr(config, "dropout_rate", args.dropout_rate)
    config.num_mlp_layers = getattr(config, "num_mlp_layers", args.num_mlp_layers)
    config.constraint_scale = getattr(config, "constraint_scale", args.constraint_scale)
    config.use_value_constraint = getattr(config, "use_value_constraint", True)
    config.use_tanh_activations = getattr(config, "use_tanh_activations", False)
    student = build_student(config, device)
    student.load_state_dict(ckpt["student_state_dict"])
    return student, config


def build_teacher(device: torch.device) -> nn.Module:
    teacher = resnet18_1d(num_classes=len(set(BEAT_LABEL_MAP.values())))
    return teacher.to(device)


def train_teacher(
    teacher: nn.Module,
    loaders: DatasetBundle,
    device: torch.device,
    args: argparse.Namespace,
) -> nn.Module:
    optimizer = Adam(teacher.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    base_weights = loaders.class_weights.to(device)

    best_val_macro_f1 = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.teacher_max_epochs + 1):
        teacher.train()
        running_loss = 0.0
        total = 0

        if args.weight_warmup_epochs > 0:
            warm_frac = min(1.0, epoch / float(args.weight_warmup_epochs))
            epoch_weights = torch.ones_like(base_weights) * (1 - warm_frac) + base_weights * warm_frac
        else:
            epoch_weights = base_weights.clone()
        ce_loss_fn = nn.CrossEntropyLoss(weight=epoch_weights)

        for signals, labels in loaders.train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = teacher(signals)
            loss = ce_loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        val_loss, val_metrics, _, _ = evaluate_with_probs(teacher, loaders.val_loader, device)
        gen_loss, gen_metrics, _, _ = evaluate_with_probs(teacher, loaders.gen_loader, device)
        scheduler.step(val_loss)

        print(
            f"[KD-Teacher] Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val MacroF1 {val_metrics['macro_f1']:.3f} Acc {val_metrics['accuracy']:.3f} | "
            f"Gen MacroF1 {gen_metrics['macro_f1']:.3f} Acc {gen_metrics['accuracy']:.3f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(teacher.state_dict())
            patience_counter = 0
            print("[KD-Teacher] -> New best teacher (by MacroF1)")
        else:
            patience_counter += 1
            if patience_counter >= args.teacher_patience:
                print("[KD-Teacher] Early stopping triggered")
                break

    if best_state is not None:
        teacher.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.teacher_path), exist_ok=True)
    torch.save({"teacher_state_dict": teacher.state_dict()}, args.teacher_path)
    print(f"[KD-Teacher] Saved teacher checkpoint to {args.teacher_path}")
    return teacher

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = build_dataloaders(args, device)
    student, student_config = load_student(args, device)

    print("[KD] Baseline student evaluation (pre-KD)...")
    _, base_val_metrics, _, _ = evaluate_with_probs(student, loaders.val_loader, device)
    _, base_gen_metrics, _, _ = evaluate_with_probs(student, loaders.gen_loader, device)
    print(
        f"[KD] Baseline -> Val MacroF1 {base_val_metrics['macro_f1']:.3f} Acc {base_val_metrics['accuracy']:.3f} | "
        f"Gen MacroF1 {base_gen_metrics['macro_f1']:.3f} Acc {base_gen_metrics['accuracy']:.3f}"
    )

    if os.path.exists(args.teacher_path):
        teacher_ckpt = torch.load(args.teacher_path, map_location=device)
        teacher = build_teacher(device)
        teacher.load_state_dict(teacher_ckpt["teacher_state_dict"])
        print(f"[KD] Loaded teacher checkpoint from {args.teacher_path}")
    else:
        print("[KD] No teacher checkpoint found -> training teacher first")
        teacher = build_teacher(device)
        teacher = train_teacher(teacher, loaders, device, args)

    print("[KD] Teacher evaluation ...")
    _, teacher_val_metrics, _, _ = evaluate_with_probs(teacher, loaders.val_loader, device)
    _, teacher_gen_metrics, _, _ = evaluate_with_probs(teacher, loaders.gen_loader, device)
    print(
        f"[KD] Teacher -> Val MacroF1 {teacher_val_metrics['macro_f1']:.3f} Acc {teacher_val_metrics['accuracy']:.3f} | "
        f"Gen MacroF1 {teacher_gen_metrics['macro_f1']:.3f} Acc {teacher_gen_metrics['accuracy']:.3f}"
    )

    use_kd = teacher_val_metrics["macro_f1"] > base_val_metrics["macro_f1"]
    kd_alpha = args.kd_alpha if use_kd else 0.0
    print(
        f"[KD] KD gating -> use_kd={use_kd} | kd_alpha={kd_alpha:.2f} | "
        f"reason: teacher_val_macro_f1 {teacher_val_metrics['macro_f1']:.3f} vs student {base_val_metrics['macro_f1']:.3f}"
    )

    optimizer = Adam(student.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    base_weights = loaders.class_weights.to(device)

    best_val_macro_f1 = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        teacher.eval()
        running_loss = 0.0
        total = 0
        if args.weight_warmup_epochs > 0:
            warm_frac = min(1.0, epoch / float(args.weight_warmup_epochs))
            epoch_weights = torch.ones_like(base_weights) * (1 - warm_frac) + base_weights * warm_frac
        else:
            warm_frac = 1.0
            epoch_weights = base_weights.clone()
        ce_loss_fn = nn.CrossEntropyLoss(weight=epoch_weights)
        if epoch == 1 or (args.weight_warmup_epochs > 0 and warm_frac < 1.0 and epoch % 5 == 0):
            print(
                f"[KD] Epoch {epoch}: CE weight warmup alpha={warm_frac:.2f}, weights={epoch_weights.detach().cpu().numpy()}"
            )

        for signals, labels in loaders.train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            student_logits, _ = student(signals)
            ce_loss = ce_loss_fn(student_logits, labels)
            if use_kd:
                with torch.no_grad():
                    teacher_logits, _ = teacher(signals)
                kd_loss = kd_logit_loss(student_logits, teacher_logits, temperature=args.kd_temperature)
                loss = (1 - kd_alpha) * ce_loss + kd_alpha * kd_loss
            else:
                loss = ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        val_loss, val_metrics, _, _ = evaluate_with_probs(student, loaders.val_loader, device)
        gen_loss, gen_metrics, _, _ = evaluate_with_probs(student, loaders.gen_loader, device)

        scheduler.step(val_loss)

        print(
            f"[KD] Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val MacroF1 {val_metrics['macro_f1']:.3f} Acc {val_metrics['accuracy']:.3f} | "
            f"Gen MacroF1 {gen_metrics['macro_f1']:.3f} Acc {gen_metrics['accuracy']:.3f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
            print("[KD] -> New best model (by 4-class MacroF1)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[KD] Early stopping triggered")
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    val_loss, val_metrics, val_true, val_pred = evaluate_with_probs(student, loaders.val_loader, device)
    gen_loss, gen_metrics, gen_true, gen_pred = evaluate_with_probs(student, loaders.gen_loader, device)

    print(
        f"[KD] Final -> Val MacroF1 {val_metrics['macro_f1']:.3f} Acc {val_metrics['accuracy']:.3f} | "
        f"Gen MacroF1 {gen_metrics['macro_f1']:.3f} Acc {gen_metrics['accuracy']:.3f}"
    )

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "student_KD.pth")
    torch.save({"student_state_dict": student.state_dict(), "config": vars(student_config)}, save_path)
    print(f"[KD] Saved student checkpoint to {save_path}")


if __name__ == "__main__":
    main()
