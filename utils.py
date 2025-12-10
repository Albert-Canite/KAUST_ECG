"""Training utilities for ECG classification."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_class_weights(labels: np.ndarray, abnormal_boost: float = 1.2) -> torch.Tensor:
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
    return torch.tensor(weights, dtype=torch.float32)


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


def kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)
