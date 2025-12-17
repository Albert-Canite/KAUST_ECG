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
from sklearn.metrics import auc, confusion_matrix, roc_curve

from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from utils import (
    BalancedBatchSampler,
    compute_class_weights,
    compute_multiclass_metrics,
    confusion_metrics,
    make_weighted_sampler,
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
    metrics_fn = compute_multiclass_metrics if threshold is None else confusion_metrics
    sample_debug: Optional[Dict[str, torch.Tensor]] = None
    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            prob_all = torch.softmax(logits, dim=1)
            prob_abnormal = prob_all[:, 1:].sum(dim=1)
            if threshold is None:
                pred = torch.argmax(logits, dim=1)
            else:
                pred = (prob_abnormal >= threshold).long()
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            if return_probs:
                probs.extend(prob_abnormal.cpu().tolist())
            if sample_debug is None:
                sample_debug = {
                    "y_true_4": labels.detach().cpu(),
                    "p_abnormal": prob_abnormal.detach().cpu(),
                }
                sample_pred = (prob_abnormal >= (0.5 if threshold is None else threshold)).long()
                sample_debug["pred_bin"] = sample_pred.detach().cpu()
                sample_debug["y_true_bin"] = (labels != 0).long().detach().cpu()
    avg_loss = total_loss / max(total, 1)
    if threshold is None:
        metrics = metrics_fn(trues, preds, num_classes)  # type: ignore[arg-type]
    else:
        y_true_bin = [int(y != 0) for y in trues]
        metrics = metrics_fn(y_true_bin, preds)

    if return_probs and sample_debug is not None:
        unique_y = torch.unique(torch.tensor(trues))
        y_true_bin_tensor = (torch.tensor(trues) != 0).long()
        pos = int((y_true_bin_tensor == 1).sum())
        neg = int((y_true_bin_tensor == 0).sum())
        print(f"[Eval] Unique y_true (4-class): {unique_y.tolist()} | bin pos={pos}, neg={neg}")
        if threshold is None:
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
            "y_true_4=", sample_debug["y_true_4"][:10].tolist(),
            "y_true_bin=", sample_debug["y_true_bin"][:10].tolist(),
            "p_abnormal=", sample_debug["p_abnormal"][:10].tolist(),
            "pred_bin=", sample_debug["pred_bin"][:10].tolist(),
        )

    return avg_loss, metrics, trues, preds, probs


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Backward-compatible boolean flags with --name / --no-name."""

    parser.add_argument(f"--{name}", dest=name, action="store_true", help=f"Enable {help_text}")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT-BIH ECG training with cross-entropy baseline")
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
    parser.add_argument("--class_weight_abnormal", type=float, default=1.35)
    parser.add_argument("--class_weight_max_ratio", type=float, default=2.0)
    parser.add_argument("--generalization_score_weight", type=float, default=0.35)
    parser.add_argument("--threshold_target_miss", type=float, default=0.10)
    parser.add_argument("--threshold_max_fpr", type=float, default=0.10)
    parser.add_argument(
        "--threshold_recall_gain",
        type=float,
        default=2.0,
        help="Sensitivity gain when scoring thresholds to prefer lower miss rates",
    )
    parser.add_argument(
        "--threshold_miss_penalty",
        type=float,
        default=1.25,
        help="Penalty weight on miss rate during blended threshold scoring",
    )
    parser.add_argument(
        "--threshold_gen_recall_gain",
        type=float,
        default=2.5,
        help="Sensitivity gain applied to generalization metrics during threshold sweeps",
    )
    parser.add_argument(
        "--threshold_gen_miss_penalty",
        type=float,
        default=1.35,
        help="Miss-rate penalty applied to generalization metrics during threshold sweeps",
    )
    _add_bool_arg(
        parser,
        "use_fn_penalty_4cls",
        default=False,
        help_text="binary-style FN penalty on abnormal beats (4-class debug default: off)",
    )
    _add_bool_arg(
        parser,
        "use_fn_penalty",
        default=True,
        help_text="adaptive abnormal upweighting based on recent miss rate (set false for plain CE)",
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
    batch_sampler = None
    # Multi-class collapse was driven by over-balancing; keep natural priors by default for 4-class.
    sampler_boost = 1.0
    if NUM_CLASSES == 2 and abnormal_ratio < 0.35:
        sampler = make_weighted_sampler(tr_y, abnormal_boost=sampler_boost)
        print(
            "Enabling weighted sampler without extra abnormal boost to avoid collapse; "
            f"boost={sampler_boost:.2f}"
        )
    elif NUM_CLASSES > 2:
        print("Skipping weighted sampler for 4-class to preserve normal/abnormal prior")
    if NUM_CLASSES == 2 and abnormal_ratio < 0.45:
        try:
            batch_sampler = BalancedBatchSampler(tr_y, batch_size=args.batch_size)
            print("Using balanced batch sampler to keep per-batch class mix stable")
        except ValueError:
            batch_sampler = None

    if batch_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
        )
    val_loader = DataLoader(ECGBeatDataset(va_x, va_y), batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(ECGBeatDataset(gen_x, gen_y), batch_size=args.batch_size, shuffle=False)

    # Keep loss gently reflecting class prior; avoid abnormal over-weighting that suppresses class N.
    effective_abnormal_boost = 1.0 if sampler is not None else min(args.class_weight_abnormal, 1.0)
    class_weights_np = compute_class_weights(
        tr_y,
        abnormal_boost=effective_abnormal_boost,
        max_ratio=args.class_weight_max_ratio,
        num_classes=NUM_CLASSES,
        power=0.5,
    )
    raw_weights = []
    for idx, count in enumerate(class_counts):
        freq = count / max(total_counts, 1)
        base = (1.0 / max(freq, 1e-8)) ** 0.5
        if idx != 0:
            base *= effective_abnormal_boost
        raw_weights.append(base)
    mean_w = float(class_weights_np.mean()) if class_weights_np.numel() > 0 else 0.0
    min_w = mean_w / args.class_weight_max_ratio if args.class_weight_max_ratio else float("nan")
    max_w = mean_w * args.class_weight_max_ratio if args.class_weight_max_ratio else float("nan")
    print(
        "Class weights computed as (1/freq)^0.5 with abnormal boost "
        f"{effective_abnormal_boost}, normalized to mean~1: raw={np.round(raw_weights, 4)}"
    )
    print(
        f"Clamped to max_ratio={args.class_weight_max_ratio}: final weights="
        f"{np.round(class_weights_np.cpu().numpy(), 4)} (mean={mean_w:.4f}, min={min_w:.4f}, max={max_w:.4f})"
    )

    class_weights = class_weights_np.to(device)
    base_weights = class_weights.clone()

    miss_ema = 0.25

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
            "data_range": [data_min, data_max],
        }
    )

    print(f"Preprocessed input range: [{data_min:.3f}, {data_max:.3f}] (expected within [-1, 1])")

    best_val_f1 = -float("inf")
    best_state = None
    best_threshold = 0.5
    patience_counter = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        running_loss = 0.0
        total = 0

        epoch_weights = base_weights.clone()
        # Binary-style FN penalty retained for experiments; disabled by default for 4-class CE.
        if args.use_fn_penalty_4cls and epoch_weights.numel() > 1:
            adaptive_pos_boost = 1.0 + miss_ema * 0.8
            for cls in range(1, NUM_CLASSES):
                epoch_weights[cls] = torch.clamp(
                    base_weights[cls] * adaptive_pos_boost,
                    min=base_weights.min() * 0.8,
                    max=base_weights.max() * 2.0,
                )
        ce_loss_fn = nn.CrossEntropyLoss(weight=epoch_weights)

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

        val_loss, val_metrics_mc, val_true, val_pred_mc, val_probs = evaluate(
            student, val_loader, device, NUM_CLASSES, return_probs=True
        )
        gen_loss, gen_metrics_mc, gen_true, gen_pred_mc, gen_probs = evaluate(
            student, gen_loader, device, NUM_CLASSES, return_probs=True
        )

        val_true_bin = [int(y != 0) for y in val_true]
        gen_true_bin = [int(y != 0) for y in gen_true]

        best_thr_epoch, val_metrics, gen_metrics = sweep_thresholds_blended(
            val_true_bin,
            val_probs,
            gen_true_bin,
            gen_probs,
            gen_weight=args.generalization_score_weight,
            recall_gain=args.threshold_recall_gain,
            miss_penalty=args.threshold_miss_penalty,
            gen_recall_gain=args.threshold_gen_recall_gain,
            gen_miss_penalty=args.threshold_gen_miss_penalty,
            miss_target=args.threshold_target_miss,
            fpr_cap=args.threshold_max_fpr,
        )

        miss_ema = 0.8 * miss_ema + 0.2 * val_metrics["miss_rate"]

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val BinF1 {val_metrics['f1']:.3f} Miss {val_metrics['miss_rate'] * 100:.2f}% FPR {val_metrics['fpr'] * 100:.2f}% "
            f"| Val MacroF1 {val_metrics_mc['macro_f1']:.3f} | "
            f"Gen BinF1 {gen_metrics['f1']:.3f} Miss {gen_metrics['miss_rate'] * 100:.2f}% FPR {gen_metrics['fpr'] * 100:.2f}% "
            f"| Gen MacroF1 {gen_metrics_mc['macro_f1']:.3f} | Thr {best_thr_epoch:.2f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_miss": val_metrics["miss_rate"],
                "val_fpr": val_metrics["fpr"],
                "val_macro_f1": val_metrics_mc["macro_f1"],
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
                "gen_macro_f1": gen_metrics_mc["macro_f1"],
                "threshold": best_thr_epoch,
            }
        )

        _write_log(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_miss": val_metrics["miss_rate"],
                "val_fpr": val_metrics["fpr"],
                "val_macro_f1": val_metrics_mc["macro_f1"],
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
                "gen_macro_f1": gen_metrics_mc["macro_f1"],
                "threshold": best_thr_epoch,
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = student.state_dict()
            best_threshold = best_thr_epoch
            patience_counter = 0
            print("  -> New best model saved.")
            _write_log(
                {
                    "event": "best",
                    "epoch": epoch,
                    "val_f1": val_metrics["f1"],
                    "val_miss": val_metrics["miss_rate"],
                    "val_fpr": val_metrics["fpr"],
                    "val_macro_f1": val_metrics_mc["macro_f1"],
                    "gen_f1": gen_metrics["f1"],
                    "gen_miss": gen_metrics["miss_rate"],
                    "gen_fpr": gen_metrics["fpr"],
                    "gen_macro_f1": gen_metrics_mc["macro_f1"],
                    "threshold": best_thr_epoch,
                }
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch >= args.min_epochs:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    val_loss, val_metrics_mc, val_true, val_pred_mc, val_probs = evaluate(
        student, val_loader, device, NUM_CLASSES, return_probs=True
    )
    gen_loss, gen_metrics_mc, gen_true, gen_pred_mc, gen_probs = evaluate(
        student, gen_loader, device, NUM_CLASSES, return_probs=True
    )

    val_true_bin = [int(y != 0) for y in val_true]
    gen_true_bin = [int(y != 0) for y in gen_true]

    best_threshold, val_metrics, gen_metrics = sweep_thresholds_blended(
        val_true_bin,
        val_probs,
        gen_true_bin,
        gen_probs,
        gen_weight=args.generalization_score_weight,
        recall_gain=args.threshold_recall_gain,
        miss_penalty=args.threshold_miss_penalty,
        gen_recall_gain=args.threshold_gen_recall_gain,
        gen_miss_penalty=args.threshold_gen_miss_penalty,
        miss_target=args.threshold_target_miss,
        fpr_cap=args.threshold_max_fpr,
    )
    val_pred_bin = (np.array(val_probs) >= best_threshold).astype(int).tolist()
    gen_pred_bin = (np.array(gen_probs) >= best_threshold).astype(int).tolist()

    print(
        f"Final Val@thr={best_threshold:.2f}: loss={val_loss:.4f}, F1={val_metrics['f1']:.3f}, "
        f"miss={val_metrics['miss_rate'] * 100:.2f}%, fpr={val_metrics['fpr'] * 100:.2f}%"
    )
    print(
        f"Generalization@thr={best_threshold:.2f}: loss={gen_loss:.4f}, F1={gen_metrics['f1']:.3f}, "
        f"miss={gen_metrics['miss_rate'] * 100:.2f}%, fpr={gen_metrics['fpr'] * 100:.2f}%"
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
            "best_threshold": best_threshold,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_miss": val_metrics["miss_rate"],
            "val_fpr": val_metrics["fpr"],
            "gen_loss": gen_loss,
            "gen_f1": gen_metrics["f1"],
            "gen_miss": gen_metrics["miss_rate"],
            "gen_fpr": gen_metrics["fpr"],
            "val_macro_f1": val_metrics_mc["macro_f1"],
            "gen_macro_f1": gen_metrics_mc["macro_f1"],
        }
    )

    # Persist probabilities for offline threshold resweeps and diagnostics
    np.save(os.path.join("artifacts", "val_probs.npy"), np.array(val_probs))
    np.save(os.path.join("artifacts", "gen_probs.npy"), np.array(gen_probs))
    np.save(os.path.join("artifacts", "val_labels.npy"), np.array(val_true))
    np.save(os.path.join("artifacts", "gen_labels.npy"), np.array(gen_true))

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
    _save_roc(val_true_bin, val_probs, "Val")
    _save_roc(gen_true_bin, gen_probs, "Generalization")
    _save_confusion(val_true, val_pred_mc, "Val_4class", list(range(NUM_CLASSES)), CLASS_NAMES)
    _save_confusion(gen_true, gen_pred_mc, "Generalization_4class", list(range(NUM_CLASSES)), CLASS_NAMES)
    _save_confusion(val_true_bin, val_pred_bin, "Val_binary", [0, 1], ["Normal", "Abnormal"])
    _save_confusion(gen_true_bin, gen_pred_bin, "Generalization_binary", [0, 1], ["Normal", "Abnormal"])
    print("Saved training curves, ROC curves, and confusion matrices to ./artifacts")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
