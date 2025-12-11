"""Training script for segment-aware ECG classification with plain cross-entropy."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


@dataclass
class KDConfig:
    use_kd: bool
    start_epoch: int
    alpha: float
    temperature: float
    ema_decay: float
    warmup_epochs: int
    confidence_floor: float
    confidence_power: float


def kd_logit_loss(logits_student: torch.Tensor, logits_teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = torch.log_softmax(logits_student / temperature, dim=1)
    teacher_probs = torch.softmax(logits_teacher / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)


def kd_alpha_schedule(base_alpha: float, current_epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if current_epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return base_alpha
    progress = min(1.0, float(current_epoch - start_epoch + 1) / float(warmup_epochs))
    return base_alpha * progress


def update_teacher(student: nn.Module, teacher: nn.Module, ema_decay: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data.mul_(ema_decay).add_(p_s.data, alpha=1.0 - ema_decay)
        for b_t, b_s in zip(teacher.buffers(), student.buffers()):
            b_t.copy_(b_s)


def build_teacher(args: argparse.Namespace, device: torch.device, student: nn.Module) -> nn.Module:
    teacher = build_student(args, device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


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
    parser.add_argument("--seed", type=int, default=42)
    _add_bool_arg(parser, "use_value_constraint", default=True, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")
    _add_bool_arg(parser, "use_kd", default=True, help_text="logit-level knowledge distillation with EMA teacher")
    parser.add_argument("--kd_start_epoch", type=int, default=12, help="Epoch to start KD and EMA teacher updates")
    parser.add_argument("--kd_alpha", type=float, default=0.03, help="Weight for KD loss blending")
    parser.add_argument("--kd_temperature", type=float, default=3.0, help="Temperature for KD softening")
    parser.add_argument("--ema_decay", type=float, default=0.997, help="EMA decay for teacher parameter updates")
    parser.add_argument(
        "--kd_warmup_epochs",
        type=int,
        default=20,
        help="Number of epochs to linearly ramp KD alpha after start_epoch",
    )
    parser.add_argument(
        "--kd_confidence_floor",
        type=float,
        default=0.70,
        help="Teacher confidence floor before applying KD weight (0 disables gating)",
    )
    parser.add_argument(
        "--kd_confidence_power",
        type=float,
        default=2.0,
        help="Exponent for confidence-based KD gating (higher=more selective)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    kd_config = KDConfig(
        use_kd=args.use_kd,
        start_epoch=args.kd_start_epoch,
        alpha=args.kd_alpha,
        temperature=args.kd_temperature,
        ema_decay=args.ema_decay,
        warmup_epochs=args.kd_warmup_epochs,
        confidence_floor=args.kd_confidence_floor,
        confidence_power=args.kd_confidence_power,
    )

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

    sampler = None
    batch_sampler = None
    sampler_boost = 1.2
    if abnormal_ratio < 0.35:
        sampler = make_weighted_sampler(tr_y, abnormal_boost=sampler_boost)
        print(
            "Enabling mild abnormal oversampling: "
            f"boost={sampler_boost:.2f}, expected abnormal frac≈{min(0.5, abnormal_ratio * sampler_boost):.2f}"
        )
    if abnormal_ratio < 0.45:
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

    # Mitigate collapse to the majority class by balancing cross-entropy with class weights
    class_counts = np.bincount(tr_y, minlength=2)
    class_weights_np = compute_class_weights(
        tr_y,
        abnormal_boost=args.class_weight_abnormal,
        max_ratio=args.class_weight_max_ratio,
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
    teacher = build_teacher(args, device, student)

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
        teacher.eval()
        effective_alpha_epoch = kd_alpha_schedule(
            kd_config.alpha, epoch, kd_config.start_epoch, kd_config.warmup_epochs
        )
        kd_active_epoch = kd_config.use_kd and epoch >= kd_config.start_epoch
        avg_kd_alpha = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0
        running_total_loss = 0.0
        total = 0

        adaptive_pos_boost = 1.0 + miss_ema * 0.8
        epoch_weights = base_weights.clone()
        epoch_weights[1] = torch.clamp(
            base_weights[1] * adaptive_pos_boost,
            min=base_weights.min() * 0.8,
            max=base_weights.max() * 2.0,
        )
        ce_loss_fn = nn.CrossEntropyLoss(weight=epoch_weights)

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, _ = student(signals)
            ce_loss = ce_loss_fn(logits, labels)
            kd_loss = torch.tensor(0.0, device=device)
            loss = ce_loss
            batch_kd_alpha = 0.0

            if kd_active_epoch:
                with torch.no_grad():
                    logits_teacher, _ = teacher(signals)
                kd_loss = kd_logit_loss(logits, logits_teacher, kd_config.temperature)
                if kd_config.confidence_floor > 0.0:
                    teacher_probs = torch.softmax(logits_teacher / kd_config.temperature, dim=1)
                    teacher_conf = teacher_probs.max(dim=1).values.mean().item()
                    confidence_scale = max(
                        0.0,
                        min(1.0, (teacher_conf - kd_config.confidence_floor) / (1.0 - kd_config.confidence_floor)),
                    ) ** kd_config.confidence_power
                    batch_kd_alpha = effective_alpha_epoch * confidence_scale
                else:
                    batch_kd_alpha = effective_alpha_epoch
                loss = (1.0 - batch_kd_alpha) * ce_loss + batch_kd_alpha * kd_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            if kd_active_epoch:
                update_teacher(student, teacher, kd_config.ema_decay)

            batch_size = labels.size(0)
            running_ce_loss += ce_loss.item() * batch_size
            running_kd_loss += kd_loss.item() * batch_size
            running_total_loss += loss.item() * batch_size
            total += labels.size(0)
            avg_kd_alpha += batch_kd_alpha * batch_size

        train_loss = running_total_loss / max(total, 1)
        train_ce_loss = running_ce_loss / max(total, 1)
        train_kd_loss = running_kd_loss / max(total, 1)
        epoch_avg_kd_alpha = avg_kd_alpha / max(total, 1)

        val_loss, _, val_true, _, val_probs = evaluate(
            student, val_loader, device, return_probs=True
        )
        gen_loss, _, gen_true, _, gen_probs = evaluate(
            student, gen_loader, device, return_probs=True
        )

        best_thr_epoch, val_metrics, gen_metrics = sweep_thresholds_blended(
            val_true,
            val_probs,
            gen_true,
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
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} (CE {train_ce_loss:.4f}, KD {train_kd_loss:.4f}) | "
            f"ValLoss {val_loss:.4f} | "
            f"Val F1 {val_metrics['f1']:.3f} Miss {val_metrics['miss_rate'] * 100:.2f}% FPR {val_metrics['fpr'] * 100:.2f}% | "
            f"Gen F1 {gen_metrics['f1']:.3f} Miss {gen_metrics['miss_rate'] * 100:.2f}% FPR {gen_metrics['fpr'] * 100:.2f}% | Thr {best_thr_epoch:.2f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ce_loss": train_ce_loss,
                "train_kd_loss": train_kd_loss,
                "kd_alpha_effective": epoch_avg_kd_alpha if kd_active_epoch else 0.0,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_miss": val_metrics["miss_rate"],
                "val_fpr": val_metrics["fpr"],
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
                "threshold": best_thr_epoch,
            }
        )

        _write_log(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ce_loss": train_ce_loss,
                "train_kd_loss": train_kd_loss,
                "kd_alpha_effective": epoch_avg_kd_alpha if kd_active_epoch else 0.0,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_miss": val_metrics["miss_rate"],
                "val_fpr": val_metrics["fpr"],
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
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
                    "gen_f1": gen_metrics["f1"],
                    "gen_miss": gen_metrics["miss_rate"],
                    "gen_fpr": gen_metrics["fpr"],
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

    val_loss, _, val_true, _, val_probs = evaluate(
        student, val_loader, device, return_probs=True
    )
    gen_loss, _, gen_true, _, gen_probs = evaluate(
        student, gen_loader, device, return_probs=True
    )

    best_threshold, val_metrics, gen_metrics = sweep_thresholds_blended(
        val_true,
        val_probs,
        gen_true,
        gen_probs,
        gen_weight=args.generalization_score_weight,
        recall_gain=args.threshold_recall_gain,
        miss_penalty=args.threshold_miss_penalty,
        gen_recall_gain=args.threshold_gen_recall_gain,
        gen_miss_penalty=args.threshold_gen_miss_penalty,
        miss_target=args.threshold_target_miss,
        fpr_cap=args.threshold_max_fpr,
    )
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
        }
    )

    # Persist probabilities for offline threshold resweeps and diagnostics
    np.save(os.path.join("artifacts", "val_probs.npy"), np.array(val_probs))
    np.save(os.path.join("artifacts", "gen_probs.npy"), np.array(gen_probs))
    np.save(os.path.join("artifacts", "val_labels.npy"), np.array(val_true))
    np.save(os.path.join("artifacts", "gen_labels.npy"), np.array(gen_true))

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "student_model.pth")
    checkpoint = {
        "student_state_dict": student.state_dict(),
        "config": vars(args),
        "best_threshold": best_threshold,
    }
    if kd_config.use_kd:
        checkpoint["teacher_state_dict"] = teacher.state_dict()
    torch.save(checkpoint, save_path)
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
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
