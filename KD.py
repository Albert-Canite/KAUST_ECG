"""Knowledge distillation script for MIT-BIH ECG student model."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from train import GENERALIZATION_RECORDS, TRAIN_RECORDS, build_student
from utils import BalancedBatchSampler, compute_class_weights, confusion_metrics, make_weighted_sampler, sweep_thresholds_blended

# Distillation hyperparameters
KD_TEMPERATURE = 3.0
KD_ALPHA = 0.1  # weight for logit-level KD
KD_BETA = 0.05  # weight for feature-level KD
KD_FEATURE_DIM = 32
KD_LR = 5e-4
KD_EPOCHS = 30
KD_FREEZE_ENCODER = True
KD_MISS_WEIGHT = 1.35  # extra weight on positives to curb miss rate


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Backward-compatible boolean flags with --name / --no-name."""

    parser.add_argument(f"--{name}", dest=name, action="store_true", help=f"Enable {help_text}")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge distillation for ECG student model")
    parser.add_argument("--data_path", type=str, default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_mlp_layers", type=int, default=3)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--class_weight_abnormal", type=float, default=1.35)
    parser.add_argument("--class_weight_max_ratio", type=float, default=2.0)
    parser.add_argument("--generalization_score_weight", type=float, default=0.35)
    parser.add_argument("--threshold_target_miss", type=float, default=0.10)
    parser.add_argument("--threshold_max_fpr", type=float, default=0.12)
    parser.add_argument("--threshold_recall_gain", type=float, default=2.0)
    parser.add_argument("--threshold_miss_penalty", type=float, default=1.25)
    parser.add_argument("--threshold_gen_recall_gain", type=float, default=2.5)
    parser.add_argument("--threshold_gen_miss_penalty", type=float, default=1.35)
    _add_bool_arg(parser, "use_value_constraint", default=True, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")
    parser.add_argument("--teacher_path", type=str, default=os.path.join("saved_models", "teacher_model.pth"))
    parser.add_argument("--student_path", type=str, default=os.path.join("saved_models", "student_model.pth"))
    parser.add_argument("--kd_output", type=str, default=os.path.join("saved_models", "student_KD.pth"))
    parser.add_argument("--skip_teacher_train", action="store_true", help="Skip teacher training if checkpoint missing")
    return parser.parse_args()


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    gen_loader: DataLoader
    val_arrays: Tuple[np.ndarray, np.ndarray]
    gen_arrays: Tuple[np.ndarray, np.ndarray]
    class_weights: torch.Tensor


def build_dataloaders(args: argparse.Namespace, device: torch.device) -> DatasetBundle:
    train_x, train_y = load_records(TRAIN_RECORDS, args.data_path)
    gen_x, gen_y = load_records(GENERALIZATION_RECORDS, args.data_path)
    if train_x.size == 0 or gen_x.size == 0:
        raise RuntimeError("No data loaded. Check data path and wfdb installation.")

    tr_x, tr_y, va_x, va_y = split_dataset(train_x, train_y, val_ratio=0.2)

    train_dataset = ECGBeatDataset(tr_x, tr_y)
    sampler = None
    batch_sampler = None
    abnormal_ratio = float(np.mean(tr_y)) if len(tr_y) > 0 else 0.0
    sampler_boost = 1.2
    if abnormal_ratio < 0.35:
        sampler = make_weighted_sampler(tr_y, abnormal_boost=sampler_boost)
    if abnormal_ratio < 0.45:
        try:
            batch_sampler = BalancedBatchSampler(tr_y, batch_size=args.batch_size)
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
    class_weights = compute_class_weights(tr_y, abnormal_boost=args.class_weight_abnormal, max_ratio=args.class_weight_max_ratio).to(
        device
    )
    return DatasetBundle(train_loader, val_loader, gen_loader, (va_x, va_y), (gen_x, gen_y), class_weights)


def hydrate_student_args(base_args: argparse.Namespace, student_config: argparse.Namespace) -> argparse.Namespace:
    """Ensure KD args carry student build hyperparameters even if not passed via CLI."""

    merged = argparse.Namespace(**vars(base_args))
    for field, default in [
        ("dropout_rate", base_args.dropout_rate),
        ("num_mlp_layers", base_args.num_mlp_layers),
        ("constraint_scale", base_args.constraint_scale),
        ("class_weight_abnormal", base_args.class_weight_abnormal),
        ("class_weight_max_ratio", base_args.class_weight_max_ratio),
        ("generalization_score_weight", base_args.generalization_score_weight),
        ("threshold_target_miss", base_args.threshold_target_miss),
        ("threshold_max_fpr", base_args.threshold_max_fpr),
        ("threshold_recall_gain", base_args.threshold_recall_gain),
        ("threshold_miss_penalty", base_args.threshold_miss_penalty),
        ("threshold_gen_recall_gain", base_args.threshold_gen_recall_gain),
        ("threshold_gen_miss_penalty", base_args.threshold_gen_miss_penalty),
        ("use_value_constraint", True),
        ("use_tanh_activations", False),
    ]:
        setattr(merged, field, getattr(student_config, field, default))
    return merged


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D18(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = self.global_pool(out).squeeze(-1)
        logits = self.fc(features)
        return logits, features


def build_teacher(args: argparse.Namespace, device: torch.device) -> nn.Module:
    teacher = ResNet1D18(num_classes=len(set(BEAT_LABEL_MAP.values()))).to(device)
    return teacher


def kd_logit_loss(logits_s: torch.Tensor, logits_t: torch.Tensor, temperature: float) -> torch.Tensor:
    p_t = F.softmax(logits_t / temperature, dim=1)
    log_p_s = F.log_softmax(logits_s / temperature, dim=1)
    loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)
    return loss


def project_features(feat: torch.Tensor, projector: nn.Module) -> torch.Tensor:
    proj = projector(feat)
    proj = F.normalize(proj, dim=1)
    return proj


def evaluate_with_probs(
    model: nn.Module, data_loader: DataLoader, device: torch.device, threshold: float = 0.5
) -> Tuple[float, Dict[str, float], List[int], List[float]]:
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
            pred = (prob_pos >= threshold).long()
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            probs.extend(prob_pos.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    metrics = confusion_metrics(trues, preds)
    try:
        fpr, tpr, _ = roc_curve(trues, probs)
        metrics["roc_auc"] = auc(fpr, tpr)
    except ValueError:
        metrics["roc_auc"] = 0.5
    return avg_loss, metrics, trues, probs


def train_teacher(args: argparse.Namespace, dataloaders: DatasetBundle, device: torch.device) -> nn.Module:
    teacher = build_teacher(args, device)
    optimizer = Adam(teacher.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        teacher.train()
        running_loss = 0.0
        total = 0
        for signals, labels in dataloaders.train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = teacher(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        train_loss = running_loss / max(total, 1)

        val_loss, val_metrics, _, _ = evaluate_with_probs(teacher, dataloaders.val_loader, device)
        scheduler.step(val_loss)
        print(
            f"[Teacher] Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val ROC {val_metrics['roc_auc']:.3f} | Miss {val_metrics['miss_rate']*100:.2f}% | FPR {val_metrics['fpr']*100:.2f}%"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = teacher.state_dict()
            patience_counter = 0
            print("  -> New best teacher checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Teacher early stopping")
                break

    if best_state is not None:
        teacher.load_state_dict(best_state)

    torch.save({"state_dict": teacher.state_dict()}, args.teacher_path)
    print(f"Saved teacher to {args.teacher_path}")
    return teacher


def load_student(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, float, argparse.Namespace]:
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
    threshold = float(ckpt.get("best_threshold", 0.5))
    return student, threshold, config


def baseline_evaluation(model: nn.Module, loaders: DatasetBundle, device: torch.device, threshold: float) -> Dict[str, object]:
    val_loss, val_metrics, val_true, val_probs = evaluate_with_probs(model, loaders.val_loader, device, threshold)
    gen_loss, gen_metrics, gen_true, gen_probs = evaluate_with_probs(model, loaders.gen_loader, device, threshold)

    print(
        f"Baseline Student -> Val ROC {val_metrics['roc_auc']:.3f}, Miss {val_metrics['miss_rate']*100:.2f}% FPR {val_metrics['fpr']*100:.2f}% | "
        f"Gen ROC {gen_metrics['roc_auc']:.3f}, Miss {gen_metrics['miss_rate']*100:.2f}% FPR {gen_metrics['fpr']*100:.2f}%"
    )
    return {
        "val": (val_loss, val_metrics, val_true, val_probs),
        "gen": (gen_loss, gen_metrics, gen_true, gen_probs),
    }


def distill_student(
    args: argparse.Namespace, loaders: DatasetBundle, teacher: nn.Module, device: torch.device, init_student: nn.Module
) -> Tuple[nn.Module, Dict[str, object]]:
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_student(args, device)
    student.load_state_dict(init_student.state_dict())

    # Optionally freeze encoder layers (convs)
    if KD_FREEZE_ENCODER:
        for name, param in student.named_parameters():
            if any(prefix in name for prefix in ["conv_p", "conv_qrs", "conv_t", "conv_global"]):
                param.requires_grad = False

    projector_s = nn.Linear(4, KD_FEATURE_DIM).to(device)
    projector_t = nn.Linear(256, KD_FEATURE_DIM).to(device)

    params = list(filter(lambda p: p.requires_grad, student.parameters())) + list(projector_s.parameters()) + list(projector_t.parameters())
    optimizer = Adam(params, lr=KD_LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    class_weights = loaders.class_weights.clone()
    if class_weights.numel() >= 2:
        class_weights[1] = class_weights[1] * KD_MISS_WEIGHT
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    history: List[Dict[str, float]] = []
    best_state = None
    best_miss = float("inf")
    best_threshold = 0.5

    for epoch in range(1, KD_EPOCHS + 1):
        student.train()
        running_loss = 0.0
        total = 0
        for signals, labels in loaders.train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits_s, feat_s = student(signals)
            with torch.no_grad():
                logits_t, feat_t = teacher(signals)
            loss_ce = ce_loss(logits_s, labels)
            loss_kd = kd_logit_loss(logits_s, logits_t, KD_TEMPERATURE)
            proj_s = project_features(feat_s, projector_s)
            proj_t = project_features(feat_t, projector_t)
            loss_feat = F.mse_loss(proj_s, proj_t)
            loss = loss_ce + KD_ALPHA * loss_kd + KD_BETA * loss_feat
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        train_loss = running_loss / max(total, 1)
        val_loss, val_metrics, val_true, val_probs = evaluate_with_probs(student, loaders.val_loader, device)
        gen_loss, gen_metrics, gen_true, gen_probs = evaluate_with_probs(student, loaders.gen_loader, device)

        thr, val_at_thr, gen_at_thr = sweep_thresholds_blended(
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

        blended_miss = (1 - args.generalization_score_weight) * val_at_thr["miss_rate"] + args.generalization_score_weight * gen_at_thr["miss_rate"]
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_at_thr.get("f1", 0.0),
                "val_miss": val_at_thr.get("miss_rate", 0.0),
                "val_fpr": val_at_thr.get("fpr", 0.0),
                "gen_f1": gen_at_thr.get("f1", 0.0),
                "gen_miss": gen_at_thr.get("miss_rate", 0.0),
                "gen_fpr": gen_at_thr.get("fpr", 0.0),
                "threshold": thr,
            }
        )
        print(
            f"[KD] Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val F1 {val_at_thr['f1']:.3f} Miss {val_at_thr['miss_rate']*100:.2f}% FPR {val_at_thr['fpr']*100:.2f}% | "
            f"Gen F1 {gen_at_thr['f1']:.3f} Miss {gen_at_thr['miss_rate']*100:.2f}% FPR {gen_at_thr['fpr']*100:.2f}% | Thr {thr:.2f}"
        )

        if blended_miss < best_miss:
            best_miss = blended_miss
            best_state = student.state_dict()
            best_threshold = thr

    if best_state is not None:
        student.load_state_dict(best_state)

    torch.save({"student_state_dict": student.state_dict(), "best_threshold": best_threshold}, args.kd_output)
    print(f"Saved distilled student to {args.kd_output}")
    return student, {"history": history, "best_threshold": best_threshold}


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[float]]:
    model.eval()
    trues: List[int] = []
    probs: List[float] = []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            logits, _ = model(signals)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            trues.extend(labels.tolist())
            probs.extend(prob_pos.cpu().tolist())
    return trues, probs


def plot_roc_comparison(val_true: List[int], baseline_probs: List[float], teacher_probs: List[float], kd_probs: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    fpr_s, tpr_s, _ = roc_curve(val_true, baseline_probs)
    fpr_t, tpr_t, _ = roc_curve(val_true, teacher_probs)
    fpr_k, tpr_k, _ = roc_curve(val_true, kd_probs)
    auc_s = auc(fpr_s, tpr_s)
    auc_t = auc(fpr_t, tpr_t)
    auc_k = auc(fpr_k, tpr_k)
    plt.plot(fpr_s, tpr_s, label=f"Student (baseline) AUC={auc_s:.3f}")
    plt.plot(fpr_t, tpr_t, label=f"Teacher AUC={auc_t:.3f}")
    plt.plot(fpr_k, tpr_k, label=f"Student (KD) AUC={auc_k:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = build_dataloaders(args, device)

    student_base, base_threshold, student_config = load_student(args, device)
    args = hydrate_student_args(args, student_config)
    baseline = baseline_evaluation(student_base, loaders, device, base_threshold)

    # Teacher handling
    if os.path.exists(args.teacher_path):
        teacher = build_teacher(args, device)
        ckpt_t = torch.load(args.teacher_path, map_location=device)
        teacher.load_state_dict(ckpt_t.get("state_dict", ckpt_t))
        print(f"Loaded teacher from {args.teacher_path}")
    else:
        if args.skip_teacher_train:
            raise FileNotFoundError("Teacher checkpoint missing and training skipped.")
        print("Teacher checkpoint missing. Training teacher...")
        teacher = train_teacher(args, loaders, device)
        # Evaluate teacher after training
        val_loss_t, val_metrics_t, _, _ = evaluate_with_probs(teacher, loaders.val_loader, device)
        gen_loss_t, gen_metrics_t, _, _ = evaluate_with_probs(teacher, loaders.gen_loader, device)
        print(
            f"Teacher -> Val loss {val_loss_t:.4f} Miss {val_metrics_t['miss_rate']*100:.2f}% FPR {val_metrics_t['fpr']*100:.2f}% | "
            f"Gen Miss {gen_metrics_t['miss_rate']*100:.2f}% FPR {gen_metrics_t['fpr']*100:.2f}%"
        )

    # Distillation
    kd_student, kd_info = distill_student(args, loaders, teacher, device, student_base)

    # Collect probabilities for comparison on validation set
    val_true, base_probs = baseline["val"][2], baseline["val"][3]
    _, teacher_probs = collect_probs(teacher, loaders.val_loader, device)
    _, kd_probs = collect_probs(kd_student, loaders.val_loader, device)
    plot_roc_comparison(val_true, base_probs, teacher_probs, kd_probs, os.path.join("figures", "kd_roc_comparison.png"))
    print("Saved ROC comparison to ./figures/kd_roc_comparison.png")

    # Evaluate distilled student
    kd_threshold = kd_info.get("best_threshold", 0.5)
    kd_val_loss, kd_val_metrics, _, _ = evaluate_with_probs(kd_student, loaders.val_loader, device, threshold=kd_threshold)
    kd_gen_loss, kd_gen_metrics, _, _ = evaluate_with_probs(kd_student, loaders.gen_loader, device, threshold=kd_threshold)

    print("\nPerformance Summary (Val set):")
    print(
        f"  Baseline Student -> Miss {baseline['val'][1]['miss_rate']*100:.2f}% | FPR {baseline['val'][1]['fpr']*100:.2f}%"
    )
    teacher_val_loss, teacher_val_metrics, _, _ = evaluate_with_probs(teacher, loaders.val_loader, device)
    print(
        f"  Teacher -> Loss {teacher_val_loss:.4f} Miss {teacher_val_metrics['miss_rate']*100:.2f}% | FPR {teacher_val_metrics['fpr']*100:.2f}%"
    )
    print(
        f"  Student (KD) -> Loss {kd_val_loss:.4f} Miss {kd_val_metrics['miss_rate']*100:.2f}% | FPR {kd_val_metrics['fpr']*100:.2f}%"
        f" | Thr {kd_threshold:.2f}"
    )

    print("\nPerformance Summary (Generalization set):")
    print(
        f"  Baseline Student -> Miss {baseline['gen'][1]['miss_rate']*100:.2f}% | FPR {baseline['gen'][1]['fpr']*100:.2f}%"
    )
    teacher_gen_loss, teacher_gen_metrics, _, _ = evaluate_with_probs(teacher, loaders.gen_loader, device)
    print(
        f"  Teacher -> Loss {teacher_gen_loss:.4f} Miss {teacher_gen_metrics['miss_rate']*100:.2f}% | FPR {teacher_gen_metrics['fpr']*100:.2f}%"
    )
    print(
        f"  Student (KD) -> Loss {kd_gen_loss:.4f} Miss {kd_gen_metrics['miss_rate']*100:.2f}% | FPR {kd_gen_metrics['fpr']*100:.2f}%"
        f" | Thr {kd_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
