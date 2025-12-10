"""Training utilities for ECG classification."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(
    labels: np.ndarray,
    abnormal_boost: float = 1.5,
    max_ratio: float = 3.0,
) -> torch.Tensor:
    """Inverse-frequency class weights with optional abnormal boost and ratio clamp."""

    counter = Counter(labels.tolist())
    total = len(labels)
    weights: List[float] = []
    num_classes = len(set(labels.tolist()))
    for i in range(num_classes):
        if i in counter:
            base = total / (num_classes * counter[i])
            if num_classes == 2 and i == 1:
                base *= abnormal_boost
            weights.append(base)
        else:
            weights.append(1.0)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    # Clamp extreme ratios to avoid collapsing to predicting全异常或全正常
    if max_ratio is not None and max_ratio > 0:
        mean_w = weight_tensor.mean()
        min_w = mean_w / max_ratio
        max_w = mean_w * max_ratio
        weight_tensor = weight_tensor.clamp(min=min_w, max=max_w)

    return weight_tensor


def confusion_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    miss_rate = fn / (tp + fn + 1e-8)
    fpr = fp / (tn + fp + 1e-8)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "miss_rate": miss_rate,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def sweep_thresholds(
    y_true: List[int], probs: List[float], thresholds: List[float] | None = None
) -> Tuple[float, Dict[str, float]]:
    """Search for the best decision threshold using a composite score.

    The score mirrors training early-stopping logic: ``f1 - miss_rate - fpr``.

    Args:
        y_true: Ground-truth labels (0/1).
        probs: Predicted probabilities for the positive class.
        thresholds: Optional list of thresholds to evaluate. If None, use a
            coarse linspace in [0.05, 0.95].

    Returns:
        best_threshold, best_metrics (dict from ``confusion_metrics``)
    """

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, num=19).tolist()

    y_true_arr = np.array(y_true)
    probs_arr = np.array(probs)
    best_score = -float("inf")
    best_thr = 0.5
    best_metrics: Dict[str, float] = {}

    for thr in thresholds:
        preds = (probs_arr >= thr).astype(int).tolist()
        metrics = confusion_metrics(y_true_arr.tolist(), preds)
        score = metrics["f1"] - metrics["miss_rate"] - metrics["fpr"]
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


def kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


def make_weighted_sampler(labels: np.ndarray, abnormal_boost: float = 1.0) -> WeightedRandomSampler:
    """Create a weighted sampler to upsample minority/abnormal beats.

    Args:
        labels: Array of integer labels.
        abnormal_boost: Multiplicative boost for the abnormal class (label==1).

    Returns:
        WeightedRandomSampler configured with per-sample weights.
    """

    label_list = labels.tolist()
    counts = Counter(label_list)
    num_samples = len(label_list)
    class_weights: Dict[int, float] = {}
    for cls, cnt in counts.items():
        # inverse frequency weighting
        class_weights[cls] = num_samples / (len(counts) * cnt)
    if 1 in class_weights:
        class_weights[1] *= abnormal_boost
    sample_weights = [class_weights[y] for y in label_list]
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
