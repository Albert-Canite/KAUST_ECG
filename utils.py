"""Training utilities for ECG classification."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_class_weights(
    labels: np.ndarray,
    max_ratio: float = 5.0,
    num_classes: int | None = None,
    power: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency weights normalized near mean=1 for multi-class tasks.

    Default settings use full inverse frequency (power=1.0) with a modest clamp
    (max_ratio=5) so rare S/V beats receive noticeably larger weights without
    overwhelming the normal prior. A separate warmup in the training loop can
    blend these weights in gradually to avoid instability early in training.
    """

    counter = Counter(labels.tolist())
    total = len(labels)
    weights: List[float] = []
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1 if len(labels) > 0 else 0

    eps = 1e-8
    for i in range(num_classes):
        freq = counter.get(i, 0) / max(total, 1)
        base = (1.0 / max(freq, eps)) ** power
        weights.append(base)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    mean_w = weight_tensor.mean().clamp(min=eps)
    weight_tensor = weight_tensor / mean_w

    if max_ratio is not None and max_ratio > 0:
        mean_w = weight_tensor.mean()
        min_w = mean_w / max_ratio
        max_w = mean_w * max_ratio
        weight_tensor = weight_tensor.clamp(min=min_w, max=max_w)

    return weight_tensor


def compute_multiclass_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {
        i: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for i, (p, r, f) in enumerate(zip(precision, recall, f1))
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


def make_weighted_sampler(labels: np.ndarray, power: float = 0.5) -> WeightedRandomSampler:
    """Create a weighted sampler to softly upsample minority beats for 4-class training."""

    label_list = labels.tolist()
    counts = Counter(label_list)
    num_samples = len(label_list)
    class_weights: Dict[int, float] = {}
    abnormal_labels = [lbl for lbl in counts.keys() if lbl != 0]
    for cls, cnt in counts.items():
        class_weights[cls] = (num_samples / (len(counts) * cnt)) ** power
    sample_weights = [class_weights[y] for y in label_list]
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
