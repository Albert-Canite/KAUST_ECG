"""Training utilities for ECG classification."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_class_weights(
    labels: np.ndarray,
    max_ratio: float | None = 5.0,
    num_classes: int | None = None,
    power: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency weights anchored to the majority class.

    The most frequent class is assigned weight ``1`` and rarer classes receive
    larger weights proportional to ``(max_count / count) ** power``. We clamp
    weights to ``[1, max_ratio]`` (when ``max_ratio`` is provided) so that
    majority classes are never downweighted, preventing the optimizer from
    collapsing into minority-only predictions as seen in previous runs.
    """

    counter = Counter(labels.tolist())
    weights: List[float] = []
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1 if len(labels) > 0 else 0

    eps = 1e-8
    max_count = max((counter.get(i, 0) for i in range(num_classes)), default=0)
    if max_count == 0:
        return torch.tensor([], dtype=torch.float32)

    for i in range(num_classes):
        cnt = counter.get(i, 0)
        base = (max_count / max(cnt, eps)) ** power
        weights.append(base)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    # Preserve majority weight at 1 while limiting extreme ratios to avoid
    # exploding gradients on ultra-rare classes.
    if max_ratio is not None and max_ratio > 0:
        weight_tensor = weight_tensor.clamp(min=1.0, max=max_ratio)
    else:
        weight_tensor = weight_tensor.clamp(min=1.0)

    return weight_tensor


def compute_multiclass_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {
        i: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for i, (p, r, f) in enumerate(zip(precision, recall, f1))
    }

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Highlight minority classes two ways:
    #   1) abnormal_macro_f1 averages all non-N classes (S/V/O)
    #   2) rare_macro_f1 focuses only on S/V, so O cannot mask failures on S/V
    abnormal_indices = [i for i in range(num_classes) if i != 0]
    abnormal_present = [idx for idx in abnormal_indices if idx in per_class]
    if abnormal_present:
        abnormal_f1s = [per_class[idx]["f1"] for idx in abnormal_present]
        abnormal_macro_f1 = float(np.mean(abnormal_f1s))
    else:
        abnormal_macro_f1 = macro_f1

    rare_indices = [1, 2] if num_classes >= 3 else []
    rare_present = [idx for idx in rare_indices if idx in per_class]
    if rare_present:
        rare_f1s = [per_class[idx]["f1"] for idx in rare_present]
        rare_macro_f1 = float(np.mean(rare_f1s))
    else:
        rare_macro_f1 = abnormal_macro_f1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1,
        "abnormal_macro_f1": abnormal_macro_f1,
        "rare_macro_f1": rare_macro_f1,
        "per_class": per_class,
    }


def kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


def make_weighted_sampler(
    labels: np.ndarray,
    power: float = 0.5,
    max_ratio: float | None = 5.0,
    target_mix: Optional[List[float]] = None,
    num_classes: Optional[int] = None,
) -> tuple[WeightedRandomSampler, Dict[int, float], Dict[int, float]]:
    """Create a weighted sampler to softly upsample minority beats for 4-class training.

    ``target_mix`` lets us steer the expected batch proportions directly. When provided,
    per-class weights are chosen so that ``weight * count`` matches the target mix
    (normalized). This gently lifts S/V exposure without forcing perfect balance and still
    respects ``max_ratio`` to prevent runaway oversampling.

    ``max_ratio`` prevents ultra-rare classes from exploding the sampling weights, which
    previously led to models collapsing to predict a single minority class. We normalize
    weights so that the smallest class weight is 1.0, then clamp larger classes.
    """

    label_list = labels.tolist()
    counts = Counter(label_list)
    num_samples = len(label_list)
    raw_weights: Dict[int, float] = {}

    if num_classes is None and target_mix is not None:
        num_classes = len(target_mix)
    if num_classes is None:
        num_classes = max(counts.keys(), default=-1) + 1

    if target_mix is not None and any(target_mix):
        mix = np.array(target_mix, dtype=float)
        if mix.shape[0] != num_classes:
            raise ValueError(
                f"target_mix length {mix.shape[0]} does not match num_classes {num_classes}"
            )
        if mix.sum() <= 0:
            raise ValueError("target_mix must sum to a positive value")
        mix = mix / mix.sum()

        for cls in range(num_classes):
            cnt = counts.get(cls, 0)
            if cnt == 0:
                # No samples for this class -> cannot resample it.
                continue
            observed_frac = cnt / float(num_samples)
            raw_weights[cls] = (mix[cls] / max(observed_frac, 1e-8)) ** 1.0
    else:
        for cls, cnt in counts.items():
            raw_weights[cls] = (num_samples / (len(counts) * cnt)) ** power

    if not raw_weights:
        sampler = WeightedRandomSampler([], num_samples=0)
        return sampler, {}, {}

    min_w = min(raw_weights.values())
    scaled_weights = {cls: w / max(min_w, 1e-8) for cls, w in raw_weights.items()}

    if max_ratio is not None and max_ratio > 0:
        scaled_weights = {cls: min(w, max_ratio) for cls, w in scaled_weights.items()}

    # Expected sampling proportion for diagnostics: proportional to weight * count.
    denom = sum(scaled_weights.get(c, 0.0) * counts.get(c, 0) for c in range(num_classes))
    if denom > 0:
        sampler_mix = {
            cls: float(scaled_weights.get(cls, 0.0) * counts.get(cls, 0) / denom)
            for cls in range(num_classes)
        }
    else:
        sampler_mix = {cls: 0.0 for cls in range(num_classes)}

    sample_weights = [scaled_weights[y] for y in label_list]
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
    return sampler, scaled_weights, sampler_mix


class FocalLoss(torch.nn.Module):
    """Multi-class focal loss with optional per-class alpha."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        if self.alpha is not None:
            alpha_factor = (self.alpha + 1e-8)[targets]
        else:
            alpha_factor = torch.ones_like(targets, dtype=log_probs.dtype, device=logits.device)

        focal_weight = alpha_factor * torch.pow(1.0 - torch.sum(probs * targets_one_hot, dim=1), self.gamma)
        ce_loss = -torch.sum(targets_one_hot * log_probs, dim=1)
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
