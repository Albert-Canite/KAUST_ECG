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
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from train import GENERALIZATION_RECORDS, TRAIN_RECORDS, build_student
from utils import (
    BalancedBatchSampler,
    compute_class_weights,
    compute_multiclass_metrics,
    confusion_metrics,
    make_weighted_sampler,
    sweep_thresholds_blended,
)

# Distillation hyperparameters
KD_TEMPERATURE = 3.0
KD_ALPHA = 0.1  # weight for logit-level KD
KD_BETA = 0.05  # weight for feature-level KD
KD_FEATURE_DIM = 32
KD_LR = 5e-4
KD_EPOCHS = 30
KD_FREEZE_ENCODER = True
KD_MISS_WEIGHT = 1.35  # extra weight on positives to curb miss rate
KD_POS_MARGIN = 0.6  # encourage positive logits to exceed this probability
KD_POS_PENALTY = 0.35  # scale for miss-focused auxiliary penalty
KD_TEACHER_GAP_TOL = 0.05  # if teacher overfits, down-weight its logits
KD_TEACHER_CONF = 0.65  # only trust teacher KD when it is confident
KD_TEACHER_GEN_MARGIN = 0.0  # disable KD if teacher miss exceeds student gen miss by this margin


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
    parser.add_argument(
        "--kd_gen_mix_ratio",
        type=float,
        default=0.35,
        help="Fraction of generalization samples to mix into KD training (0-1)",
    )
    return parser.parse_args()


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    kd_loader: DataLoader
    val_loader: DataLoader
    gen_loader: DataLoader
    val_arrays: Tuple[np.ndarray, np.ndarray]
    gen_arrays: Tuple[np.ndarray, np.ndarray]
    class_weights: torch.Tensor
    kd_class_weights: torch.Tensor


def build_dataloaders(args: argparse.Namespace, device: torch.device) -> DatasetBundle:
    train_x, train_y = load_records(TRAIN_RECORDS, args.data_path)
    gen_x, gen_y = load_records(GENERALIZATION_RECORDS, args.data_path)
    if train_x.size == 0 or gen_x.size == 0:
        raise RuntimeError("No data loaded. Check data path and wfdb installation.")

    tr_x, tr_y, va_x, va_y = split_dataset(train_x, train_y, val_ratio=0.2)

    train_dataset = ECGBeatDataset(tr_x, tr_y)
    sampler = None
    batch_sampler = None
    num_classes = len(set(BEAT_LABEL_MAP.values()))
    class_counts = np.bincount(tr_y, minlength=num_classes)
    total_counts = class_counts.sum()
    abnormal_ratio = 1.0 - (class_counts[0] / total_counts) if total_counts > 0 else 0.0
    print(f"[KD] Train class counts (N,S,V,O): {class_counts.tolist()} | total={int(total_counts)}")
    sampler_boost = 1.0
    if num_classes == 2 and abnormal_ratio < 0.35:
        sampler = make_weighted_sampler(tr_y, abnormal_boost=sampler_boost)
    elif num_classes > 2:
        print("[KD] Skipping weighted sampler for 4-class to preserve normal/abnormal prior")
    if len(np.unique(tr_y)) == 2 and abnormal_ratio < 0.45:
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
    effective_abnormal_boost = 1.0 if sampler is not None else min(args.class_weight_abnormal, 1.0)
    class_weights = compute_class_weights(
        tr_y,
        abnormal_boost=effective_abnormal_boost,
        max_ratio=args.class_weight_max_ratio,
        num_classes=num_classes,
        power=0.5,
    ).to(device)
    raw_weights = []
    for idx, count in enumerate(class_counts):
        freq = count / max(total_counts, 1)
        base = (1.0 / max(freq, 1e-8)) ** 0.5
        if idx != 0:
            base *= effective_abnormal_boost
        raw_weights.append(base)
    print(
        "[KD] Class weights (1/freq)^0.5 with abnormal boost "
        f"{effective_abnormal_boost}: raw={np.round(raw_weights, 4)} | final={np.round(class_weights.cpu().numpy(), 4)}"
    )

    # Build a KD loader that mixes in part of the generalization distribution to reduce miss drift
    if args.kd_gen_mix_ratio > 0:
        mix_k = int(len(gen_x) * args.kd_gen_mix_ratio)
        kd_x = np.concatenate([tr_x, gen_x[:mix_k]], axis=0)
        kd_y = np.concatenate([tr_y, gen_y[:mix_k]], axis=0)
    else:
        kd_x, kd_y = tr_x, tr_y
    kd_dataset = ECGBeatDataset(kd_x, kd_y)
    kd_sampler = None
    kd_batch_sampler = None
    kd_abnormal_ratio = float(np.mean(kd_y)) if len(kd_y) > 0 else 0.0
    if num_classes == 2 and kd_abnormal_ratio < 0.35:
        kd_sampler = make_weighted_sampler(kd_y, abnormal_boost=sampler_boost)
    elif num_classes > 2:
        print("[KD] Skipping KD sampler for 4-class to preserve normal/abnormal prior")
    if kd_abnormal_ratio < 0.45:
        try:
            kd_batch_sampler = BalancedBatchSampler(kd_y, batch_size=args.batch_size)
        except ValueError:
            kd_batch_sampler = None
    if kd_batch_sampler is not None:
        kd_loader = DataLoader(kd_dataset, batch_sampler=kd_batch_sampler)
    else:
        kd_loader = DataLoader(
            kd_dataset,
            batch_size=args.batch_size,
            shuffle=kd_sampler is None,
            sampler=kd_sampler,
        )
    kd_abnormal_boost = 1.0 if kd_sampler is not None else min(args.class_weight_abnormal * KD_MISS_WEIGHT, 1.0)
    kd_class_weights = compute_class_weights(
        kd_y,
        abnormal_boost=kd_abnormal_boost,
        max_ratio=args.class_weight_max_ratio,
        num_classes=num_classes,
        power=0.5,
    ).to(device)

    return DatasetBundle(
        train_loader,
        kd_loader,
        val_loader,
        gen_loader,
        (va_x, va_y),
        (gen_x, gen_y),
        class_weights,
        kd_class_weights,
    )


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
    preds_mc: List[int] = []
    trues_bin: List[int] = []
    trues_4: List[int] = []
    probs: List[float] = []
    sample_debug: Dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            prob_all = torch.softmax(logits, dim=1)
            prob_abnormal = prob_all[:, 1:].sum(dim=1)
            pred = (prob_abnormal >= threshold).long()
            pred_mc = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            preds_mc.extend(pred_mc.cpu().tolist())
            trues_4.extend(labels.cpu().tolist())
            trues_bin.extend((labels != 0).long().cpu().tolist())
            probs.extend(prob_abnormal.cpu().tolist())
            if sample_debug is None:
                sample_debug = {
                    "y_true_4": labels.detach().cpu(),
                    "y_true_bin": (labels != 0).long().detach().cpu(),
                    "p_abnormal": prob_abnormal.detach().cpu(),
                    "pred_bin": pred.detach().cpu(),
                }
    avg_loss = total_loss / max(total, 1)

    unique_y = torch.unique(torch.tensor(trues_4))
    y_true_bin_tensor = torch.tensor(trues_bin)
    pos = int((y_true_bin_tensor == 1).sum())
    neg = int((y_true_bin_tensor == 0).sum())
    print(f"[Eval/KD] Unique y_true (4-class): {unique_y.tolist()} | bin pos={pos}, neg={neg}")

    metrics = confusion_metrics(trues_bin, preds)
    mc_metrics = compute_multiclass_metrics(trues_4, preds_mc, num_classes=len(set(BEAT_LABEL_MAP.values())))
    metrics["macro_f1_4"] = mc_metrics["macro_f1"]
    metrics["per_class_4"] = mc_metrics["per_class"]
    if len(torch.unique(y_true_bin_tensor)) < 2:
        metrics["roc_auc"] = 0.5
        print("[Eval/KD] ROC skipped due to single-class labels; defaulting to 0.5")
    else:
        fpr, tpr, _ = roc_curve(trues_bin, probs)
        metrics["roc_auc"] = auc(fpr, tpr)
    cm = confusion_matrix(trues_4, preds_mc, labels=list(range(len(set(BEAT_LABEL_MAP.values())))))
    print(f"[Eval/KD] 4-class confusion matrix (rows=true, cols=pred):\n{cm}")
    for cid, cname in enumerate(["N", "S", "V", "O"]):
        mc = mc_metrics["per_class"].get(cid, {})
        print(
            f"[Eval/KD] Class {cname}: precision={mc.get('precision', 0):.3f} "
            f"recall={mc.get('recall', 0):.3f} f1={mc.get('f1', 0):.3f}"
        )

    if sample_debug is not None:
        print(
            "[Eval/KD] Sample sanity (first batch, first 10):",
            "y_true_4=", sample_debug["y_true_4"][:10].tolist(),
            "y_true_bin=", sample_debug["y_true_bin"][:10].tolist(),
            "p_abnormal=", sample_debug["p_abnormal"][:10].tolist(),
            "pred_bin=", sample_debug["pred_bin"][:10].tolist(),
            f"| ROC={metrics['roc_auc']:.3f} miss={metrics['miss_rate']:.3f} fpr={metrics['fpr']:.3f}",
        )

    return avg_loss, metrics, trues_bin, probs


def train_teacher(args: argparse.Namespace, dataloaders: DatasetBundle, device: torch.device) -> nn.Module:
    teacher = build_teacher(args, device)
    optimizer = Adam(teacher.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=dataloaders.class_weights)
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
    args: argparse.Namespace,
    loaders: DatasetBundle,
    teacher: nn.Module,
    device: torch.device,
    init_student: nn.Module,
    kd_alpha: float = KD_ALPHA,
    kd_beta: float = KD_BETA,
    use_kd: bool = True,
    teacher_threshold: float = 0.5,
) -> Tuple[nn.Module, Dict[str, object]]:
    if teacher is not None:
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
    class_weights = loaders.kd_class_weights.clone()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    history: List[Dict[str, float]] = []
    best_state = None
    best_miss = float("inf")
    best_threshold = 0.5

    for epoch in range(1, KD_EPOCHS + 1):
        student.train()
        running_loss = 0.0
        total = 0
        for signals, labels in loaders.kd_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits_s, feat_s = student(signals)
            logits_t = None
            feat_t = None
            teacher_mask = None
            if use_kd and (kd_alpha > 0 or kd_beta > 0) and teacher is not None:
                with torch.no_grad():
                    logits_t, feat_t = teacher(signals)
                    probs_t = torch.softmax(logits_t, dim=1)
                    conf_t, pred_t = probs_t.max(dim=1)
                    pred_bin = (probs_t[:, 1] >= teacher_threshold).long()
                    teacher_mask = (pred_bin == labels) & (conf_t >= KD_TEACHER_CONF)
            loss_ce = ce_loss(logits_s, labels)
            loss_kd = torch.tensor(0.0, device=device)
            loss_feat = torch.tensor(0.0, device=device)
            if use_kd and logits_t is not None and teacher_mask is not None and teacher_mask.any():
                loss_kd = kd_logit_loss(logits_s[teacher_mask], logits_t[teacher_mask], KD_TEMPERATURE)
            if use_kd and feat_t is not None and teacher_mask is not None and teacher_mask.any():
                proj_s = project_features(feat_s[teacher_mask], projector_s)
                proj_t = project_features(feat_t[teacher_mask], projector_t)
                loss_feat = F.mse_loss(proj_s, proj_t)
            prob_pos = torch.softmax(logits_s, dim=1)[:, 1]
            pos_mask = labels == 1
            miss_focus = torch.tensor(0.0, device=device)
            if pos_mask.any():
                miss_focus = F.relu(KD_POS_MARGIN - prob_pos[pos_mask]).mean()
            loss = loss_ce + kd_alpha * loss_kd + kd_beta * loss_feat + KD_POS_PENALTY * miss_focus
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
        blended_fpr = (1 - args.generalization_score_weight) * val_at_thr["fpr"] + args.generalization_score_weight * gen_at_thr["fpr"]
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

        miss_fpr_score = blended_miss + 0.35 * blended_fpr
        if miss_fpr_score < best_miss:
            best_miss = miss_fpr_score
            best_state = student.state_dict()
            best_threshold = thr

    if best_state is not None:
        student.load_state_dict(best_state)

    os.makedirs("artifacts", exist_ok=True)
    log_path = os.path.join("artifacts", "kd_training_log.csv")
    with open(log_path, "w") as f:
        f.write(
            "epoch,train_loss,val_loss,val_f1,val_miss,val_fpr,gen_f1,gen_miss,gen_fpr,threshold\n"
        )
        for row in history:
            f.write(
                f"{row['epoch']},{row['train_loss']:.6f},{row['val_loss']:.6f},{row['val_f1']:.6f},{row['val_miss']:.6f},"
                f"{row['val_fpr']:.6f},{row['gen_f1']:.6f},{row['gen_miss']:.6f},{row['gen_fpr']:.6f},{row['threshold']:.4f}\n"
            )
    print(f"Saved KD training log to {log_path}")

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


def plot_roc_comparison(
    val_true: List[int],
    val_probs: Tuple[List[float], List[float], List[float]],
    gen_true: List[int],
    gen_probs: Tuple[List[float], List[float], List[float]],
    out_path: str,
) -> None:
    plt.figure(figsize=(10, 4))

    val_labels = ["Student (baseline)", "Teacher", "Student (KD)"]
    gen_labels = val_labels
    plt.subplot(1, 2, 1)
    for probs, label in zip(val_probs, val_labels):
        fpr, tpr, _ = roc_curve(val_true, probs)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Val ROC Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    for probs, label in zip(gen_probs, gen_labels):
        fpr, tpr, _ = roc_curve(gen_true, probs)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Gen ROC Comparison")
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

    # Evaluate teacher, sweep thresholds, and gate KD based on gen performance vs student baseline
    teacher_val_loss, teacher_val_metrics, teacher_val_true, teacher_val_probs = evaluate_with_probs(
        teacher, loaders.val_loader, device
    )
    teacher_gen_loss, teacher_gen_metrics, teacher_gen_true, teacher_gen_probs = evaluate_with_probs(
        teacher, loaders.gen_loader, device
    )

    teacher_thr, teacher_val_thr_metrics, teacher_gen_thr_metrics = sweep_thresholds_blended(
        teacher_val_true,
        teacher_val_probs,
        teacher_gen_true,
        teacher_gen_probs,
        gen_weight=args.generalization_score_weight,
        recall_gain=args.threshold_recall_gain,
        miss_penalty=args.threshold_miss_penalty,
        gen_recall_gain=args.threshold_gen_recall_gain,
        gen_miss_penalty=args.threshold_gen_miss_penalty,
        miss_target=args.threshold_target_miss,
        fpr_cap=args.threshold_max_fpr,
    )

    student_gen_miss = baseline["gen"][1]["miss_rate"]
    student_gen_fpr = baseline["gen"][1]["fpr"]
    teacher_gen_miss_at_thr = teacher_gen_thr_metrics["miss_rate"]
    teacher_gen_fpr_at_thr = teacher_gen_thr_metrics["fpr"]

    kd_alpha_eff = KD_ALPHA
    kd_beta_eff = KD_BETA
    use_kd = True

    if teacher_gen_miss_at_thr >= student_gen_miss or teacher_gen_fpr_at_thr > max(
        student_gen_fpr, args.threshold_max_fpr
    ):
        kd_alpha_eff = 0.0
        kd_beta_eff = 0.0
        use_kd = False
        print(
            "Teacher underperforms student on generalization miss/FPR thresholds; disabling KD (CE fine-tune only)."
        )
    else:
        print(
            f"Teacher accepted for KD (thr={teacher_thr:.2f}) with Gen Miss {teacher_gen_miss_at_thr*100:.2f}% / FPR {teacher_gen_fpr_at_thr*100:.2f}%"
        )

    print(
        f"Teacher -> Val loss {teacher_val_loss:.4f} Miss {teacher_val_metrics['miss_rate']*100:.2f}% FPR {teacher_val_metrics['fpr']*100:.2f}% | "
        f"Gen Miss {teacher_gen_metrics['miss_rate']*100:.2f}% FPR {teacher_gen_metrics['fpr']*100:.2f}%"
    )
    print(
        f"Teacher threshold (blended) {teacher_thr:.2f} -> Val Miss {teacher_val_thr_metrics['miss_rate']*100:.2f}% FPR {teacher_val_thr_metrics['fpr']*100:.2f}% | "
        f"Gen Miss {teacher_gen_thr_metrics['miss_rate']*100:.2f}% FPR {teacher_gen_thr_metrics['fpr']*100:.2f}%"
    )

    # Distillation
    kd_student, kd_info = distill_student(
        args,
        loaders,
        teacher,
        device,
        student_base,
        kd_alpha=kd_alpha_eff,
        kd_beta=kd_beta_eff,
        use_kd=use_kd,
        teacher_threshold=teacher_thr,
    )

    # Collect probabilities for comparison on validation and generalization sets
    val_true, base_val_probs = baseline["val"][2], baseline["val"][3]
    gen_true, base_gen_probs = baseline["gen"][2], baseline["gen"][3]
    _, teacher_val_probs = collect_probs(teacher, loaders.val_loader, device)
    _, teacher_gen_probs = collect_probs(teacher, loaders.gen_loader, device)
    _, kd_val_probs = collect_probs(kd_student, loaders.val_loader, device)
    _, kd_gen_probs = collect_probs(kd_student, loaders.gen_loader, device)
    plot_roc_comparison(
        val_true,
        (base_val_probs, teacher_val_probs, kd_val_probs),
        gen_true,
        (base_gen_probs, teacher_gen_probs, kd_gen_probs),
        os.path.join("figures", "kd_roc_comparison.png"),
    )
    print("Saved ROC comparison (val/gen) to ./figures/kd_roc_comparison.png")

    # Evaluate distilled student
    kd_threshold = kd_info.get("best_threshold", 0.5)
    kd_val_loss, kd_val_metrics, _, _ = evaluate_with_probs(kd_student, loaders.val_loader, device, threshold=kd_threshold)
    kd_gen_loss, kd_gen_metrics, _, _ = evaluate_with_probs(kd_student, loaders.gen_loader, device, threshold=kd_threshold)

    print("\nPerformance Summary (Val set):")
    print(
        f"  Baseline Student -> Miss {baseline['val'][1]['miss_rate']*100:.2f}% | FPR {baseline['val'][1]['fpr']*100:.2f}% | Thr {base_threshold:.2f}"
    )
    print(
        f"  Teacher -> Loss {teacher_val_loss:.4f} Miss {teacher_val_thr_metrics['miss_rate']*100:.2f}% | FPR {teacher_val_thr_metrics['fpr']*100:.2f}% | Thr {teacher_thr:.2f}"
    )
    print(
        f"  Student (KD) -> Loss {kd_val_loss:.4f} Miss {kd_val_metrics['miss_rate']*100:.2f}% | FPR {kd_val_metrics['fpr']*100:.2f}%"
        f" | Thr {kd_threshold:.2f}"
    )

    print("\nPerformance Summary (Generalization set):")
    print(
        f"  Baseline Student -> Miss {baseline['gen'][1]['miss_rate']*100:.2f}% | FPR {baseline['gen'][1]['fpr']*100:.2f}% | Thr {base_threshold:.2f}"
    )
    teacher_gen_loss, teacher_gen_metrics, _, _ = evaluate_with_probs(teacher, loaders.gen_loader, device, threshold=teacher_thr)
    print(
        f"  Teacher -> Loss {teacher_gen_loss:.4f} Miss {teacher_gen_metrics['miss_rate']*100:.2f}% | FPR {teacher_gen_metrics['fpr']*100:.2f}% | Thr {teacher_thr:.2f}"
    )
    print(
        f"  Student (KD) -> Loss {kd_gen_loss:.4f} Miss {kd_gen_metrics['miss_rate']*100:.2f}% | FPR {kd_gen_metrics['fpr']*100:.2f}%"
        f" | Thr {kd_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
