"""Training script for segment-aware ECG classification with plain cross-entropy."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
from utils import compute_class_weights, compute_multiclass_metrics, make_weighted_sampler


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
    sample_debug: Optional[Dict[str, torch.Tensor]] = None
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
            if sample_debug is None:
                sample_debug = {
                    "y_true": labels.detach().cpu(),
                    "pred": pred.detach().cpu(),
                    "prob_all": prob_all.detach().cpu(),
                }
    avg_loss = total_loss / max(total, 1)
    metrics = compute_multiclass_metrics(trues, preds, num_classes)  # type: ignore[arg-type]

    if sample_debug is not None:
        unique_y = torch.unique(torch.tensor(trues))
        print(f"[Eval] Unique y_true (4-class): {unique_y.tolist()}")
        cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))
        print(f"[Eval] 4-class confusion matrix (rows=true, cols=pred):\n{cm}")
        per_cls = metrics.get("per_class", {}) if isinstance(metrics, dict) else {}
        for cid in range(num_classes):
            mc = per_cls.get(cid, {})
            print(
                f"[Eval] Class {CLASS_NAMES[cid]}: precision={mc.get('precision', 0):.3f} "
                f"recall={mc.get('recall', 0):.3f} f1={mc.get('f1', 0):.3f}"
            )
        print(
            "[Eval] Sample sanity (first batch, first 10):",
            "y_true=", sample_debug["y_true"][:10].tolist(),
            "pred=", sample_debug["pred"][:10].tolist(),
            "prob_first_class=", sample_debug["prob_all"][:10, 0].tolist(),
        )

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
        default=1.0,
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
        default=True,
        help_text="enable weighted sampler (sqrt balancing) for long-tail classes; disable to use class weights only",
    )
    parser.add_argument(
        "--sampler_power",
        type=float,
        default=1.0,
        help="inverse-frequency exponent for sampler (0.5=sqrt, 1.0=full balance)",
    )
    _add_bool_arg(
        parser,
        "stack_sampler_with_ce",
        default=True,
        help_text="keep inverse-frequency CE weights even when the weighted sampler is enabled",
    )
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

    train_dataset = ECGBeatDataset(tr_x, tr_y)

    sampler = None
    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(tr_y, power=args.sampler_power)
        print(
            "Using weighted sampler for 4-class to surface rare S/V beats. "
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

    # Keep loss reflecting class prior with inverse-frequency weights (no binary boosts).
    class_weights_np = compute_class_weights(
        tr_y,
        max_ratio=args.class_weight_max_ratio,
        num_classes=NUM_CLASSES,
        power=args.class_weight_power,
    )
    raw_weights = []
    for idx, count in enumerate(class_counts):
        freq = count / max(total_counts, 1)
        base = (1.0 / max(freq, 1e-8)) ** args.class_weight_power
        raw_weights.append(base)
    mean_w = float(class_weights_np.mean()) if class_weights_np.numel() > 0 else 0.0
    clamping_on = args.class_weight_max_ratio is not None and args.class_weight_max_ratio > 0
    min_w = mean_w / args.class_weight_max_ratio if clamping_on else float("nan")
    max_w = mean_w * args.class_weight_max_ratio if clamping_on else float("nan")
    print(
        f"Class weights computed as (1/freq)^{args.class_weight_power:.2f} (no binary abnormal boost), normalized to mean~1: "
        f"raw={np.round(raw_weights, 4)}"
    )
    print(
        f"Clamped to max_ratio={args.class_weight_max_ratio}: final weights="
        f"{np.round(class_weights_np.cpu().numpy(), 4)} (mean={mean_w:.4f}, min={min_w:.4f}, max={max_w:.4f})"
    )

    class_weights = class_weights_np.to(device)
    # By default we **stack** CE inverse-frequency weights with the sampler to make rare classes
    # much more expensive, which helps the model escape the N/V collapse seen in previous runs.
    # Disable stacking via --no-stack-sampler-with-ce if overshooting causes instability.
    if sampler is not None and not args.stack_sampler_with_ce:
        base_weights = torch.ones_like(class_weights)
        print(
            "Sampler active -> using uniform CE weights (sampler handles imbalance). "
            f"Original inverse-freq weights retained for logging: {np.round(class_weights.cpu().numpy(), 4)}"
        )
    else:
        base_weights = class_weights.clone()
        if sampler is not None:
            print(
                "Sampler active **and** CE stacking enabled -> combining sampler upsampling with "
                f"inverse-freq CE weights: {np.round(class_weights.cpu().numpy(), 4)}"
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
            "data_range": [data_min, data_max],
        }
    )

    print(f"Preprocessed input range: [{data_min:.3f}, {data_max:.3f}] (expected within [-1, 1])")

    best_val_macro_f1 = -float("inf")
    best_state = None
    patience_counter = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        running_loss = 0.0
        total = 0

        if args.weight_warmup_epochs > 0:
            warm_frac = min(1.0, epoch / float(args.weight_warmup_epochs))
            epoch_weights = torch.ones_like(base_weights) * (1 - warm_frac) + base_weights * warm_frac
            warm_frac_display = warm_frac
        else:
            epoch_weights = base_weights.clone()
            warm_frac_display = 1.0
        ce_loss_fn = nn.CrossEntropyLoss(weight=epoch_weights)
        if epoch == 1 or (args.weight_warmup_epochs > 0 and warm_frac < 1.0 and epoch % 5 == 0):
            print(
                f"Epoch {epoch}: CE weight warmup alpha={warm_frac_display:.2f}, weights={epoch_weights.detach().cpu().numpy()}"
            )

        for signals, labels in train_loader:
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

        val_loss, val_metrics_mc, val_true, val_pred_mc, _ = evaluate(
            student, val_loader, device, NUM_CLASSES
        )
        gen_loss, gen_metrics_mc, gen_true, gen_pred_mc, _ = evaluate(
            student, gen_loader, device, NUM_CLASSES
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val MacroF1 {val_metrics_mc['macro_f1']:.3f} Acc {val_metrics_mc['accuracy']:.3f} | "
            f"Gen MacroF1 {gen_metrics_mc['macro_f1']:.3f} Acc {gen_metrics_mc['accuracy']:.3f}"
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
                "gen_macro_f1": gen_metrics_mc["macro_f1"],
                "gen_acc": gen_metrics_mc["accuracy"],
            }
        )

        # 4-class monitoring: use macro F1 to drive checkpointing/early stopping
        if val_metrics_mc["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics_mc["macro_f1"]
            best_state = student.state_dict()
            patience_counter = 0
            print("  -> New best model saved (by 4-class MacroF1).")
            _write_log(
                {
                    "event": "best",
                    "epoch": epoch,
                    "val_macro_f1": val_metrics_mc["macro_f1"],
                    "val_acc": val_metrics_mc["accuracy"],
                    "gen_macro_f1": gen_metrics_mc["macro_f1"],
                    "gen_acc": gen_metrics_mc["accuracy"],
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
