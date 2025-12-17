from __future__ import annotations

import argparse
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
from train import GENERALIZATION_RECORDS, TRAIN_RECORDS, build_student
from utils import compute_class_weights, compute_multiclass_metrics, make_weighted_sampler


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
    parser.add_argument("--class_weight_max_ratio", type=float, default=2.0)
    _add_bool_arg(parser, "use_value_constraint", default=True, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")
    _add_bool_arg(
        parser,
        "use_weighted_sampler",
        default=True,
        help_text="enable weighted sampler (sqrt balancing) for long-tail classes",
    )
    parser.add_argument("--sampler_power", type=float, default=0.5)
    parser.add_argument("--student_path", type=str, default=os.path.join("saved_models", "student_model.pth"))
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
            "[KD] Using weighted sampler for 4-class to expose rare S/V beats; CE weights stay uniform to avoid double boosts. "
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
        power=0.5,
    )
    if sampler is not None:
        print("[KD] Sampler active -> using uniform CE weights to avoid double balancing")
        class_weights = torch.ones_like(class_weights_np)
    else:
        class_weights = class_weights_np
    class_weights = class_weights.to(device)
    print(
        "[KD] Train class counts (N,S,V,O): "
        f"{class_counts.tolist()} | weights={np.round(class_weights.cpu().numpy(), 4)}"
    )

    return DatasetBundle(train_loader, val_loader, gen_loader, (va_x, va_y), (gen_x, gen_y), class_weights)


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = build_dataloaders(args, device)
    student, student_config = load_student(args, device)

    optimizer = Adam(student.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    ce_loss_fn = nn.CrossEntropyLoss(weight=loaders.class_weights.to(device))

    best_val_macro_f1 = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        running_loss = 0.0
        total = 0
        for signals, labels in loaders.train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = student(signals)
            loss = ce_loss_fn(logits, labels)
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
            best_state = student.state_dict()
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
