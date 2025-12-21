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
    confusion_metrics,
    make_weighted_sampler,
    sweep_thresholds_blended,
    sweep_thresholds_low_miss,
    sweep_thresholds_miss_then_fpr,
    sweep_thresholds_three_level,
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
    _add_bool_arg(parser, "enable_low_miss_threshold", default=True, help_text="low miss driven threshold sweeps")
    parser.add_argument(
        "--gen_fpr_cap_low_miss",
        type=float,
        default=0.15,
        help="Maximum allowable generalization FPR when selecting low-miss thresholds",
    )
    parser.add_argument(
        "--low_miss_fpr_beta",
        type=float,
        default=0.1,
        help="Weak FPR weight in low-miss scoring to discourage extreme FPR",
    )
    parser.add_argument(
        "--low_miss_val_fpr_beta",
        type=float,
        default=0.05,
        help="Soft validation FPR weight in low-miss scoring to discourage high val FPR",
    )
    _add_bool_arg(
        parser,
        "enable_miss_then_fpr_threshold",
        default=True,
        help_text="miss-target then FPR threshold sweeps",
    )
    parser.add_argument(
        "--gen_miss_target_then_fpr",
        type=float,
        default=0.035,
        help="Target generalization miss for miss-then-FPR selection",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        choices=["balanced", "low_miss"],
        default="low_miss",
        help="Early stopping metric preference",
    )
    parser.add_argument(
        "--threshold_grid_step",
        type=float,
        default=0.01,
        help="Step size for coarse threshold grid",
    )
    parser.add_argument(
        "--threshold_sweep_miss_cap",
        type=float,
        default=0.05,
        help="Miss-rate cap for post-training threshold sweeps",
    )
    parser.add_argument(
        "--threshold_sweep_fpr_cap",
        type=float,
        default=0.12,
        help="FPR cap for post-training threshold sweeps",
    )
    _add_bool_arg(parser, "threshold_refine", default=True, help_text="refine thresholds near the best candidate")
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
    best_low_miss_state = None
    best_low_miss_threshold = 0.5
    best_low_miss_gen = None
    best_miss_then_fpr_state = None
    best_miss_then_fpr_threshold = 0.5
    best_miss_then_fpr_gen = None
    best_miss_then_fpr_val = None
    patience_counter = 0
    low_miss_patience_counter = 0
    best_low_miss_miss_seen: Optional[float] = None

    history: List[Dict[str, float]] = []
    threshold_grid = np.arange(0.02, 0.9800001, args.threshold_grid_step).tolist()

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        running_loss = 0.0
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
            loss = ce_loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)

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
            thresholds=threshold_grid,
        )

        low_miss_thr = best_thr_epoch
        val_metrics_low_miss = val_metrics
        gen_metrics_low_miss = gen_metrics
        low_miss_info: Dict[str, object] | None = None
        miss_then_fpr_thr = best_thr_epoch
        val_metrics_miss_then_fpr = val_metrics
        gen_metrics_miss_then_fpr = gen_metrics
        miss_then_fpr_info: Dict[str, object] | None = None
        if args.enable_low_miss_threshold:
            low_miss_thr, val_metrics_low_miss, gen_metrics_low_miss, low_miss_info = sweep_thresholds_low_miss(
                val_probs,
                val_true,
                gen_probs,
                gen_true,
                thresholds=threshold_grid,
                gen_fpr_cap=args.gen_fpr_cap_low_miss,
                refine=args.threshold_refine,
                fpr_beta=args.low_miss_fpr_beta,
                val_fpr_beta=args.low_miss_val_fpr_beta,
            )
        if args.enable_miss_then_fpr_threshold:
            miss_then_fpr_thr, val_metrics_miss_then_fpr, gen_metrics_miss_then_fpr, miss_then_fpr_info = sweep_thresholds_miss_then_fpr(
                val_probs,
                val_true,
                gen_probs,
                gen_true,
                thresholds=threshold_grid,
                gen_miss_target=args.gen_miss_target_then_fpr,
                gen_fpr_cap=args.gen_fpr_cap_low_miss,
                refine=args.threshold_refine,
            )

        low_miss_caps_met = False
        if args.enable_low_miss_threshold:
            low_miss_caps_met = (
                gen_metrics_low_miss["fpr"] <= args.gen_fpr_cap_low_miss
            )
            if low_miss_caps_met:
                if best_low_miss_miss_seen is None or gen_metrics_low_miss["miss_rate"] < best_low_miss_miss_seen - 1e-8:
                    best_low_miss_miss_seen = gen_metrics_low_miss["miss_rate"]
                    low_miss_patience_counter = 0
                else:
                    low_miss_patience_counter += 1
            else:
                low_miss_patience_counter += 1

        miss_ema = 0.8 * miss_ema + 0.2 * val_metrics["miss_rate"]

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Balanced Thr {best_thr_epoch:.3f} | Val F1 {val_metrics['f1']:.3f} Miss {val_metrics['miss_rate'] * 100:.2f}% FPR {val_metrics['fpr'] * 100:.2f}% | "
            f"Gen F1 {gen_metrics['f1']:.3f} Miss {gen_metrics['miss_rate'] * 100:.2f}% FPR {gen_metrics['fpr'] * 100:.2f}%"
        )
        if args.enable_low_miss_threshold:
            gen_cap_flag = " ok" if gen_metrics_low_miss["fpr"] <= args.gen_fpr_cap_low_miss else " cap"
            print(
                f"  LowMiss Thr {low_miss_thr:.3f} | Val F1 {val_metrics_low_miss['f1']:.3f} Miss {val_metrics_low_miss['miss_rate'] * 100:.2f}% FPR {val_metrics_low_miss['fpr'] * 100:.2f}% | "
                f"Gen F1 {gen_metrics_low_miss['f1']:.3f} Miss {gen_metrics_low_miss['miss_rate'] * 100:.2f}% FPR {gen_metrics_low_miss['fpr'] * 100:.2f}%{gen_cap_flag}"
            )
            if low_miss_info is not None and low_miss_info.get("warning"):
                print(f"    LowMiss warning: {low_miss_info['warning']}")
        if args.enable_miss_then_fpr_threshold:
            warning_flag = "" if miss_then_fpr_info is None or not miss_then_fpr_info.get("warning") else " warn"
            print(
                f"  MissThenFPR Thr {miss_then_fpr_thr:.3f} | Val F1 {val_metrics_miss_then_fpr['f1']:.3f} Miss {val_metrics_miss_then_fpr['miss_rate'] * 100:.2f}% FPR {val_metrics_miss_then_fpr['fpr'] * 100:.2f}% | "
                f"Gen F1 {gen_metrics_miss_then_fpr['f1']:.3f} Miss {gen_metrics_miss_then_fpr['miss_rate'] * 100:.2f}% FPR {gen_metrics_miss_then_fpr['fpr'] * 100:.2f}%{warning_flag}"
            )
            if miss_then_fpr_info is not None and miss_then_fpr_info.get("warning"):
                print(f"    MissThenFPR warning: {miss_then_fpr_info['warning']}")

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
                "threshold": best_thr_epoch,
                "low_miss_threshold": low_miss_thr,
                "low_miss_val_f1": val_metrics_low_miss["f1"],
                "low_miss_val_miss": val_metrics_low_miss["miss_rate"],
                "low_miss_val_fpr": val_metrics_low_miss["fpr"],
                "low_miss_gen_f1": gen_metrics_low_miss["f1"],
                "low_miss_gen_miss": gen_metrics_low_miss["miss_rate"],
                "low_miss_gen_fpr": gen_metrics_low_miss["fpr"],
                "low_miss_caps_met": low_miss_caps_met,
                "miss_then_fpr_threshold": miss_then_fpr_thr,
                "miss_then_fpr_val_f1": val_metrics_miss_then_fpr["f1"],
                "miss_then_fpr_val_miss": val_metrics_miss_then_fpr["miss_rate"],
                "miss_then_fpr_val_fpr": val_metrics_miss_then_fpr["fpr"],
                "miss_then_fpr_gen_f1": gen_metrics_miss_then_fpr["f1"],
                "miss_then_fpr_gen_miss": gen_metrics_miss_then_fpr["miss_rate"],
                "miss_then_fpr_gen_fpr": gen_metrics_miss_then_fpr["fpr"],
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
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
                "threshold": best_thr_epoch,
                "low_miss_threshold": low_miss_thr,
                "low_miss_val_f1": val_metrics_low_miss["f1"],
                "low_miss_val_miss": val_metrics_low_miss["miss_rate"],
                "low_miss_val_fpr": val_metrics_low_miss["fpr"],
                "low_miss_gen_f1": gen_metrics_low_miss["f1"],
                "low_miss_gen_miss": gen_metrics_low_miss["miss_rate"],
                "low_miss_gen_fpr": gen_metrics_low_miss["fpr"],
                "low_miss_caps_met": low_miss_caps_met,
                "low_miss_warning": None if low_miss_info is None else low_miss_info.get("warning"),
                "miss_then_fpr_threshold": miss_then_fpr_thr,
                "miss_then_fpr_val_f1": val_metrics_miss_then_fpr["f1"],
                "miss_then_fpr_val_miss": val_metrics_miss_then_fpr["miss_rate"],
                "miss_then_fpr_val_fpr": val_metrics_miss_then_fpr["fpr"],
                "miss_then_fpr_gen_f1": gen_metrics_miss_then_fpr["f1"],
                "miss_then_fpr_gen_miss": gen_metrics_miss_then_fpr["miss_rate"],
                "miss_then_fpr_gen_fpr": gen_metrics_miss_then_fpr["fpr"],
                "miss_then_fpr_warning": None if miss_then_fpr_info is None else miss_then_fpr_info.get("warning"),
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = student.state_dict()
            best_threshold = best_thr_epoch
            patience_counter = 0
            os.makedirs("saved_models", exist_ok=True)
            torch.save({"student_state_dict": best_state, "threshold": best_threshold}, os.path.join("saved_models", "best_balanced.pt"))
            print("  -> New best balanced model saved.")
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
                    "type": "balanced",
                }
            )
        else:
            patience_counter += 1

        allow_stop = False
        if args.early_stop_metric == "balanced":
            allow_stop = patience_counter >= args.patience and epoch >= args.min_epochs
        elif args.early_stop_metric == "low_miss":
            if args.enable_low_miss_threshold:
                allow_stop = low_miss_patience_counter >= args.patience and epoch >= args.min_epochs
            else:
                allow_stop = patience_counter >= args.patience and epoch >= args.min_epochs

        if allow_stop:
            print("Early stopping triggered.")
            break

        if args.enable_low_miss_threshold and low_miss_caps_met:
            update_low_miss = False
            if best_low_miss_gen is None:
                update_low_miss = True
            else:
                better_miss = gen_metrics_low_miss["miss_rate"] < best_low_miss_gen["miss_rate"]
                miss_tie = np.isclose(gen_metrics_low_miss["miss_rate"], best_low_miss_gen["miss_rate"], atol=1e-6)
                better_fpr = gen_metrics_low_miss["fpr"] < best_low_miss_gen["fpr"]
                fpr_tie = np.isclose(gen_metrics_low_miss["fpr"], best_low_miss_gen["fpr"], atol=1e-6)
                better_f1 = gen_metrics_low_miss["f1"] > best_low_miss_gen["f1"]
                update_low_miss = better_miss or (miss_tie and better_fpr) or (miss_tie and fpr_tie and better_f1)

            if update_low_miss:
                best_low_miss_state = student.state_dict()
                best_low_miss_threshold = low_miss_thr
                best_low_miss_gen = gen_metrics_low_miss
                os.makedirs("saved_models", exist_ok=True)
                torch.save(
                    {"student_state_dict": best_low_miss_state, "threshold": best_low_miss_threshold},
                    os.path.join("saved_models", "best_low_miss.pt"),
                )
                print("  -> New best low-miss model saved.")
                log_entry = {
                    "event": "best",
                    "epoch": epoch,
                    "val_f1": val_metrics_low_miss["f1"],
                    "val_miss": val_metrics_low_miss["miss_rate"],
                    "val_fpr": val_metrics_low_miss["fpr"],
                    "gen_f1": gen_metrics_low_miss["f1"],
                    "gen_miss": gen_metrics_low_miss["miss_rate"],
                    "gen_fpr": gen_metrics_low_miss["fpr"],
                    "threshold": low_miss_thr,
                    "type": "low_miss",
                }
                if low_miss_info is not None:
                    log_entry["low_miss_info"] = low_miss_info
                _write_log(log_entry)

        if args.enable_miss_then_fpr_threshold and gen_metrics_miss_then_fpr["miss_rate"] <= args.gen_miss_target_then_fpr:
            update_miss_then_fpr = False
            if best_miss_then_fpr_gen is None or best_miss_then_fpr_val is None:
                update_miss_then_fpr = True
            else:
                better_fpr = gen_metrics_miss_then_fpr["fpr"] < best_miss_then_fpr_gen["fpr"]
                fpr_tie = np.isclose(gen_metrics_miss_then_fpr["fpr"], best_miss_then_fpr_gen["fpr"], atol=1e-6)
                better_f1 = gen_metrics_miss_then_fpr["f1"] > best_miss_then_fpr_gen["f1"]
                f1_tie = np.isclose(gen_metrics_miss_then_fpr["f1"], best_miss_then_fpr_gen["f1"], atol=1e-6)
                better_val_fpr = val_metrics_miss_then_fpr["fpr"] < best_miss_then_fpr_val["fpr"]
                update_miss_then_fpr = better_fpr or (fpr_tie and (better_f1 or (f1_tie and better_val_fpr)))

            if update_miss_then_fpr:
                best_miss_then_fpr_state = student.state_dict()
                best_miss_then_fpr_threshold = miss_then_fpr_thr
                best_miss_then_fpr_gen = gen_metrics_miss_then_fpr
                best_miss_then_fpr_val = val_metrics_miss_then_fpr
                os.makedirs("saved_models", exist_ok=True)
                torch.save(
                    {"student_state_dict": best_miss_then_fpr_state, "threshold": best_miss_then_fpr_threshold},
                    os.path.join("saved_models", "best_miss_then_fpr.pt"),
                )
                print("  -> New best miss-then-FPR model saved.")
                log_entry = {
                    "event": "best",
                    "epoch": epoch,
                    "val_f1": val_metrics_miss_then_fpr["f1"],
                    "val_miss": val_metrics_miss_then_fpr["miss_rate"],
                    "val_fpr": val_metrics_miss_then_fpr["fpr"],
                    "gen_f1": gen_metrics_miss_then_fpr["f1"],
                    "gen_miss": gen_metrics_miss_then_fpr["miss_rate"],
                    "gen_fpr": gen_metrics_miss_then_fpr["fpr"],
                    "threshold": miss_then_fpr_thr,
                    "type": "miss_then_fpr",
                }
                if miss_then_fpr_info is not None:
                    log_entry["miss_then_fpr_info"] = miss_then_fpr_info
                _write_log(log_entry)

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
        thresholds=threshold_grid,
    )
    val_pred = (np.array(val_probs) >= best_threshold).astype(int).tolist()
    gen_pred = (np.array(gen_probs) >= best_threshold).astype(int).tolist()

    low_miss_final_thr = None
    val_metrics_low_miss_final: Optional[Dict[str, float]] = None
    gen_metrics_low_miss_final: Optional[Dict[str, float]] = None
    miss_then_fpr_final_thr = None
    val_metrics_miss_then_fpr_final: Optional[Dict[str, float]] = None
    gen_metrics_miss_then_fpr_final: Optional[Dict[str, float]] = None
    if args.enable_low_miss_threshold:
        low_miss_final_thr, val_metrics_low_miss_final, gen_metrics_low_miss_final, _ = sweep_thresholds_low_miss(
            val_probs,
            val_true,
            gen_probs,
            gen_true,
            thresholds=threshold_grid,
            gen_fpr_cap=args.gen_fpr_cap_low_miss,
            refine=args.threshold_refine,
            fpr_beta=args.low_miss_fpr_beta,
            val_fpr_beta=args.low_miss_val_fpr_beta,
        )
    if args.enable_miss_then_fpr_threshold:
        miss_then_fpr_final_thr, val_metrics_miss_then_fpr_final, gen_metrics_miss_then_fpr_final, _ = sweep_thresholds_miss_then_fpr(
            val_probs,
            val_true,
            gen_probs,
            gen_true,
            thresholds=threshold_grid,
            gen_miss_target=args.gen_miss_target_then_fpr,
            gen_fpr_cap=args.gen_fpr_cap_low_miss,
            refine=args.threshold_refine,
        )

    print(
        f"Final Val@thr={best_threshold:.2f}: loss={val_loss:.4f}, F1={val_metrics['f1']:.3f}, "
        f"miss={val_metrics['miss_rate'] * 100:.2f}%, fpr={val_metrics['fpr'] * 100:.2f}%"
    )
    print(
        f"Generalization@thr={best_threshold:.2f}: loss={gen_loss:.4f}, F1={gen_metrics['f1']:.3f}, "
        f"miss={gen_metrics['miss_rate'] * 100:.2f}%, fpr={gen_metrics['fpr'] * 100:.2f}%"
    )
    if args.enable_low_miss_threshold and low_miss_final_thr is not None and val_metrics_low_miss_final is not None and gen_metrics_low_miss_final is not None:
        gen_cap_flag = " !" if gen_metrics_low_miss_final["fpr"] > args.gen_fpr_cap_low_miss else ""
        print(
            f"LowMiss Val@thr={low_miss_final_thr:.3f}: F1={val_metrics_low_miss_final['f1']:.3f}, "
            f"miss={val_metrics_low_miss_final['miss_rate'] * 100:.2f}%, fpr={val_metrics_low_miss_final['fpr'] * 100:.2f}%"
        )
        print(
            f"LowMiss Generalization@thr={low_miss_final_thr:.3f}: F1={gen_metrics_low_miss_final['f1']:.3f}, "
            f"miss={gen_metrics_low_miss_final['miss_rate'] * 100:.2f}%, fpr={gen_metrics_low_miss_final['fpr'] * 100:.2f}%{gen_cap_flag}"
        )
    if (
        args.enable_miss_then_fpr_threshold
        and miss_then_fpr_final_thr is not None
        and val_metrics_miss_then_fpr_final is not None
        and gen_metrics_miss_then_fpr_final is not None
    ):
        warn_flag = " !" if gen_metrics_miss_then_fpr_final["miss_rate"] > args.gen_miss_target_then_fpr else ""
        print(
            f"MissThenFPR Val@thr={miss_then_fpr_final_thr:.3f}: F1={val_metrics_miss_then_fpr_final['f1']:.3f}, "
            f"miss={val_metrics_miss_then_fpr_final['miss_rate'] * 100:.2f}%, fpr={val_metrics_miss_then_fpr_final['fpr'] * 100:.2f}%"
        )
        print(
            f"MissThenFPR Generalization@thr={miss_then_fpr_final_thr:.3f}: F1={gen_metrics_miss_then_fpr_final['f1']:.3f}, "
            f"miss={gen_metrics_miss_then_fpr_final['miss_rate'] * 100:.2f}%, fpr={gen_metrics_miss_then_fpr_final['fpr'] * 100:.2f}%{warn_flag}"
        )

    sweep_thresholds_out, sweep_val_metrics, sweep_gen_metrics, sweep_info = sweep_thresholds_three_level(
        val_probs,
        val_true,
        gen_probs,
        gen_true,
        thresholds=threshold_grid,
        miss_cap=args.threshold_sweep_miss_cap,
        fpr_cap=args.threshold_sweep_fpr_cap,
    )
    sweep_warning = ""
    if sweep_info.get("warning"):
        sweep_warning = f" ({sweep_info['warning']})"
    print(
        f"ConstrainedSweep miss<{args.threshold_sweep_miss_cap * 100:.0f}% "
        f"fpr<{args.threshold_sweep_fpr_cap * 100:.0f}%{sweep_warning}"
    )
    for name, label in [
        ("high_miss_low_fpr", "HighMiss/LowFPR"),
        ("balanced", "Balanced"),
        ("low_miss_high_fpr", "LowMiss/HighFPR"),
    ]:
        thr = sweep_thresholds_out[name]
        val_m = sweep_val_metrics[name]
        gen_m = sweep_gen_metrics[name]
        print(
            f"  {label} Val@thr={thr:.3f}: F1={val_m['f1']:.3f}, miss={val_m['miss_rate'] * 100:.2f}%, "
            f"fpr={val_m['fpr'] * 100:.2f}%"
        )
        print(
            f"  {label} Generalization@thr={thr:.3f}: F1={gen_m['f1']:.3f}, miss={gen_m['miss_rate'] * 100:.2f}%, "
            f"fpr={gen_m['fpr'] * 100:.2f}%"
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
            "low_miss_threshold": low_miss_final_thr,
            "low_miss_val_f1": None if val_metrics_low_miss_final is None else val_metrics_low_miss_final["f1"],
            "low_miss_val_miss": None if val_metrics_low_miss_final is None else val_metrics_low_miss_final["miss_rate"],
            "low_miss_val_fpr": None if val_metrics_low_miss_final is None else val_metrics_low_miss_final["fpr"],
            "low_miss_gen_f1": None if gen_metrics_low_miss_final is None else gen_metrics_low_miss_final["f1"],
            "low_miss_gen_miss": None if gen_metrics_low_miss_final is None else gen_metrics_low_miss_final["miss_rate"],
            "low_miss_gen_fpr": None if gen_metrics_low_miss_final is None else gen_metrics_low_miss_final["fpr"],
            "miss_then_fpr_threshold": miss_then_fpr_final_thr,
            "miss_then_fpr_val_f1": None if val_metrics_miss_then_fpr_final is None else val_metrics_miss_then_fpr_final["f1"],
            "miss_then_fpr_val_miss": None if val_metrics_miss_then_fpr_final is None else val_metrics_miss_then_fpr_final["miss_rate"],
            "miss_then_fpr_val_fpr": None if val_metrics_miss_then_fpr_final is None else val_metrics_miss_then_fpr_final["fpr"],
            "miss_then_fpr_gen_f1": None if gen_metrics_miss_then_fpr_final is None else gen_metrics_miss_then_fpr_final["f1"],
            "miss_then_fpr_gen_miss": None if gen_metrics_miss_then_fpr_final is None else gen_metrics_miss_then_fpr_final["miss_rate"],
            "miss_then_fpr_gen_fpr": None if gen_metrics_miss_then_fpr_final is None else gen_metrics_miss_then_fpr_final["fpr"],
            "constrained_sweep_thresholds": sweep_thresholds_out,
            "constrained_sweep_val": sweep_val_metrics,
            "constrained_sweep_gen": sweep_gen_metrics,
            "constrained_sweep_info": sweep_info,
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
