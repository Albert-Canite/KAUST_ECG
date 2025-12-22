"""Training utilities for ECG classification."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, WeightedRandomSampler


def compute_class_weights(
    labels: np.ndarray,
    abnormal_boost: float = 1.35,
    max_ratio: float = 2.0,
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
    y_true: List[int],
    probs: List[float],
    thresholds: List[float] | None = None,
    miss_target: float | None = None,
    fpr_cap: float | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Search for a decision threshold with an explicit miss-rate target.

    The search first prioritizes thresholds that meet ``miss_target`` (if
    provided) and optionally satisfy an ``fpr_cap``. Among candidates that
    meet the target, the score emphasizes recall while penalizing FPR:
    ``score = f1 + 1.5 * sensitivity - fpr``. If no threshold meets the
    target, the best score across all thresholds is returned as a fallback.

    Args:
        y_true: Ground-truth labels (0/1).
        probs: Predicted probabilities for the positive class.
        thresholds: Optional list of thresholds to evaluate. If None, use a
            dense linspace in [0.05, 0.95] plus probability quantiles.
        miss_target: Optional maximum acceptable miss rate used for filtering.
        fpr_cap: Optional maximum acceptable FPR used alongside miss_target.

    Returns:
        best_threshold, best_metrics (dict from ``confusion_metrics``)
    """

    if thresholds is None:
        dense = np.linspace(0.05, 0.95, num=19)
        quantiles = np.quantile(probs, q=np.linspace(0.05, 0.95, num=19))
        thresholds = np.unique(np.concatenate([dense, quantiles])).tolist()

    y_true_arr = np.array(y_true)
    probs_arr = np.array(probs)

    def _score(metrics: Dict[str, float]) -> float:
        return metrics["f1"] + 1.5 * metrics["sensitivity"] - metrics["fpr"]

    best_score = -float("inf")
    best_thr = 0.5
    best_metrics: Dict[str, float] = {}

    filtered_candidates: List[Tuple[float, Dict[str, float]]] = []

    for thr in thresholds:
        preds = (probs_arr >= thr).astype(int).tolist()
        metrics = confusion_metrics(y_true_arr.tolist(), preds)
        if miss_target is not None and metrics["miss_rate"] <= miss_target:
            if fpr_cap is None or metrics["fpr"] <= fpr_cap:
                filtered_candidates.append((float(thr), metrics))
        score = _score(metrics)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    if filtered_candidates:
        # Choose the best among candidates that meet the constraints
        best_thr, best_metrics = max(filtered_candidates, key=lambda x: _score(x[1]))

    return best_thr, best_metrics


def sweep_thresholds_adaptive(
    y_true: List[int],
    probs: List[float],
    miss_target: float | None = None,
    fpr_cap: float | None = None,
    recall_gain: float = 1.0,
    threshold_center: float = 0.5,
    threshold_reg: float = 0.05,
    thresholds: List[float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Threshold sweep with recall emphasis and a bias toward moderate thresholds."""

    if thresholds is None:
        dense = np.linspace(0.05, 0.95, num=25)
        quantiles = np.quantile(probs, q=np.linspace(0.05, 0.95, num=13))
        thresholds = np.unique(np.concatenate([dense, quantiles])).tolist()

    y_true_arr = np.array(y_true)
    probs_arr = np.array(probs)

    def _score(metrics: Dict[str, float], thr: float) -> float:
        proximity_penalty = threshold_reg * abs(thr - threshold_center)
        return metrics["f1"] + recall_gain * metrics["sensitivity"] - metrics["fpr"] - proximity_penalty

    best_score = -float("inf")
    best_thr = threshold_center
    best_metrics: Dict[str, float] = {}

    for thr in thresholds:
        preds = (probs_arr >= thr).astype(int).tolist()
        metrics = confusion_metrics(y_true_arr.tolist(), preds)
        if miss_target is not None and metrics["miss_rate"] > miss_target:
            continue
        if fpr_cap is not None and metrics["fpr"] > fpr_cap:
            continue
        score = _score(metrics, float(thr))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    # fallback to unconstrained best
    if not best_metrics:
        for thr in thresholds:
            preds = (probs_arr >= thr).astype(int).tolist()
            metrics = confusion_metrics(y_true_arr.tolist(), preds)
            score = _score(metrics, float(thr))
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_metrics = metrics

    return best_thr, best_metrics


def sweep_thresholds_blended(
    val_true: List[int],
    val_probs: List[float],
    gen_true: List[int],
    gen_probs: List[float],
    gen_weight: float = 0.3,
    recall_gain: float = 1.5,
    miss_penalty: float = 1.0,
    gen_recall_gain: float | None = None,
    gen_miss_penalty: float | None = None,
    thresholds: List[float] | None = None,
    miss_target: float | None = None,
    fpr_cap: float | None = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Jointly sweep thresholds on val/generalization splits with blended scoring.

    Args:
        val_true: Validation labels.
        val_probs: Validation positive probabilities.
        gen_true: Generalization labels.
        gen_probs: Generalization positive probabilities.
        gen_weight: Blend weight in [0,1] for generalization when scoring.
        thresholds: Optional threshold list; defaults to dense grid + quantiles.
        recall_gain: Multiplicative weight on sensitivity to bias toward lower miss.
        miss_penalty: Penalty weight on miss rate when computing blended score.
        gen_recall_gain: Optional override on recall gain for generalization metrics.
        gen_miss_penalty: Optional override on miss penalty for generalization metrics.
        miss_target: Optional miss-rate cap applied to the blended miss.
        fpr_cap: Optional FPR cap applied to the blended FPR.

    Returns:
        best_threshold, val_metrics_at_threshold, gen_metrics_at_threshold
    """

    gen_weight = float(np.clip(gen_weight, 0.0, 1.0))
    val_recall_gain = recall_gain
    val_miss_penalty = miss_penalty
    gen_recall_gain = recall_gain if gen_recall_gain is None else gen_recall_gain
    gen_miss_penalty = miss_penalty if gen_miss_penalty is None else gen_miss_penalty
    if thresholds is None:
        dense = np.linspace(0.02, 0.98, num=21)
        quantiles = np.quantile(np.concatenate([val_probs, gen_probs]), q=np.linspace(0.05, 0.95, num=19))
        thresholds = np.unique(np.concatenate([dense, quantiles])).tolist()

    def _score(metrics: Dict[str, float], r_gain: float, m_penalty: float) -> float:
        return metrics["f1"] + r_gain * metrics["sensitivity"] - m_penalty * metrics["miss_rate"] - metrics["fpr"]

    best_score = -float("inf")
    best_thr = 0.5
    best_val: Dict[str, float] = {}
    best_gen: Dict[str, float] = {}

    for thr in thresholds:
        val_preds = (np.array(val_probs) >= thr).astype(int).tolist()
        gen_preds = (np.array(gen_probs) >= thr).astype(int).tolist()
        val_metrics = confusion_metrics(val_true, val_preds)
        gen_metrics = confusion_metrics(gen_true, gen_preds)

        blended_miss = (1 - gen_weight) * val_metrics["miss_rate"] + gen_weight * gen_metrics["miss_rate"]
        blended_fpr = (1 - gen_weight) * val_metrics["fpr"] + gen_weight * gen_metrics["fpr"]
        blended_score = (1 - gen_weight) * _score(val_metrics, val_recall_gain, val_miss_penalty) + gen_weight * _score(
            gen_metrics, gen_recall_gain, gen_miss_penalty
        )

        if miss_target is not None and blended_miss > miss_target:
            continue
        if fpr_cap is not None and blended_fpr > fpr_cap:
            continue

        if blended_score > best_score:
            best_score = blended_score
            best_thr = float(thr)
            best_val = val_metrics
            best_gen = gen_metrics

    # Fallback to the best unconstrained threshold if no candidate met constraints
    if best_val == {} or best_gen == {}:
        for thr in thresholds:
            val_preds = (np.array(val_probs) >= thr).astype(int).tolist()
            gen_preds = (np.array(gen_probs) >= thr).astype(int).tolist()
            val_metrics = confusion_metrics(val_true, val_preds)
            gen_metrics = confusion_metrics(gen_true, gen_preds)
            blended_score = (1 - gen_weight) * _score(val_metrics, val_recall_gain, val_miss_penalty) + gen_weight * _score(
                gen_metrics, gen_recall_gain, gen_miss_penalty
            )
            if blended_score > best_score:
                best_score = blended_score
                best_thr = float(thr)
                best_val = val_metrics
                best_gen = gen_metrics

    return best_thr, best_val, best_gen


def _generate_threshold_grid(
    base_step: float = 0.01, low: float = 0.02, high: float = 0.98
) -> List[float]:
    """Utility to create a uniform threshold grid within [low, high]."""

    grid = np.arange(low, high + 1e-8, base_step)
    return grid.tolist()


def sweep_thresholds_low_miss(
    val_probs: List[float],
    val_labels: List[int],
    gen_probs: List[float],
    gen_labels: List[int],
    thresholds: List[float] | None = None,
    gen_fpr_cap: float = 0.12,
    refine: bool = True,
    refine_step: float = 0.002,
    fpr_beta: float = 0.1,
    val_fpr_beta: float = 0.05,
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, object]]:
    """Sweep thresholds prioritizing lower generalization miss with soft FPR penalties.

    Selection order within candidates that satisfy gen_fpr_cap:
    1) Minimize generalization miss.
    2) Minimize generalization FPR.
    3) Minimize validation FPR (tie breaker).
    4) Maximize generalization F1.

    A weak regularizer discourages extreme FPR values while keeping miss first:
    score = -gen_miss + alpha * gen_f1 - beta_gen * gen_fpr - beta_val * val_fpr, with alpha=1.0.
    """

    if thresholds is None:
        thresholds = _generate_threshold_grid()

    val_arr = np.array(val_probs)
    gen_arr = np.array(gen_probs)

    def _eval(thr: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        val_preds = (val_arr >= thr).astype(int).tolist()
        gen_preds = (gen_arr >= thr).astype(int).tolist()
        return confusion_metrics(val_labels, val_preds), confusion_metrics(gen_labels, gen_preds)

    def _select_best(thr_list: List[float]) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, object]]:
        candidate_log = []
        fallback_best = None
        best_thr = thr_list[0]
        best_val: Dict[str, float] = {}
        best_gen: Dict[str, float] = {}
        best_record: Dict[str, object] = {}
        alpha = 1.0

        for thr in thr_list:
            val_metrics, gen_metrics = _eval(thr)
            candidate_log.append(
                {
                    "threshold": float(thr),
                    "gen_miss": gen_metrics["miss_rate"],
                    "gen_fpr": gen_metrics["fpr"],
                    "gen_f1": gen_metrics["f1"],
                    "val_fpr": val_metrics["fpr"],
                    "val_miss": val_metrics["miss_rate"],
                }
            )

            if fallback_best is None or gen_metrics["fpr"] < fallback_best[2]["fpr"] - 1e-8:
                fallback_best = (float(thr), val_metrics, gen_metrics)

            within_caps = gen_metrics["fpr"] <= gen_fpr_cap
            if within_caps:
                if best_gen == {}:
                    best_thr = float(thr)
                    best_val = val_metrics
                    best_gen = gen_metrics
                    best_record = {
                        "threshold": best_thr,
                        "gen_miss": best_gen["miss_rate"],
                        "gen_fpr": best_gen["fpr"],
                        "gen_f1": best_gen["f1"],
                        "val_fpr": best_val["fpr"],
                        "val_miss": best_val["miss_rate"],
                        "score": -best_gen["miss_rate"]
                        + alpha * best_gen["f1"]
                        - fpr_beta * best_gen["fpr"]
                        - val_fpr_beta * best_val["fpr"],
                    }
                    continue

                better_miss = gen_metrics["miss_rate"] < best_gen["miss_rate"]
                miss_tie = np.isclose(gen_metrics["miss_rate"], best_gen["miss_rate"], atol=1e-8)
                better_fpr = gen_metrics["fpr"] < best_gen["fpr"]
                fpr_tie = np.isclose(gen_metrics["fpr"], best_gen["fpr"], atol=1e-8)
                better_val_fpr = val_metrics["fpr"] < best_val["fpr"]
                val_fpr_tie = np.isclose(val_metrics["fpr"], best_val["fpr"], atol=1e-8)
                better_f1 = gen_metrics["f1"] > best_gen["f1"]
                candidate_score = -gen_metrics["miss_rate"]
                candidate_score += alpha * gen_metrics["f1"]
                candidate_score -= fpr_beta * gen_metrics["fpr"]
                candidate_score -= val_fpr_beta * val_metrics["fpr"]
                best_score = -best_gen["miss_rate"]
                best_score += alpha * best_gen["f1"]
                best_score -= fpr_beta * best_gen["fpr"]
                best_score -= val_fpr_beta * best_val["fpr"]

                if better_miss or (
                    miss_tie
                    and (
                        better_fpr
                        or (fpr_tie and (better_val_fpr or (val_fpr_tie and (better_f1 or candidate_score > best_score))))
                    )
                ):
                    best_thr = float(thr)
                    best_val = val_metrics
                    best_gen = gen_metrics
                    best_record = {
                        "threshold": best_thr,
                        "gen_miss": best_gen["miss_rate"],
                        "gen_fpr": best_gen["fpr"],
                        "gen_f1": best_gen["f1"],
                        "val_fpr": best_val["fpr"],
                        "val_miss": best_val["miss_rate"],
                        "score": candidate_score,
                    }

        if best_record == {} and fallback_best is not None:
            best_thr, best_val, best_gen = fallback_best
            best_record = {
                "threshold": best_thr,
                "gen_miss": best_gen["miss_rate"],
                "gen_fpr": best_gen["fpr"],
                "gen_f1": best_gen["f1"],
                "val_fpr": best_val["fpr"],
                "val_miss": best_val["miss_rate"],
            }

        info = {
            "gen_fpr_cap": gen_fpr_cap,
            "selection": "minimize gen miss under cap, then lower gen fpr, then lower val fpr, then higher gen f1",
            "candidates": candidate_log,
            "best_record": best_record,
        }
        return best_thr, best_val, best_gen, info

    best_thr, best_val_metrics, best_gen_metrics, info = _select_best(thresholds)
    warning = None
    if info.get("best_record", {}) == {}:
        warning = "no threshold satisfied gen fpr cap; selected minimum gen fpr"
        info["warning"] = warning
    else:
        info["warning"] = None

    if refine:
        window = 0.05
        refine_low = max(0.0, best_thr - window)
        refine_high = min(1.0, best_thr + window)
        fine_grid = np.arange(refine_low, refine_high + 1e-8, refine_step).tolist()
        best_thr, best_val_metrics, best_gen_metrics, info_refined = _select_best(fine_grid)
        info_refined["refined"] = True
        info_refined["warning"] = warning
        info = info_refined
    else:
        info["refined"] = False

    return best_thr, best_val_metrics, best_gen_metrics, info


def sweep_thresholds_miss_then_fpr(
    val_probs: List[float],
    val_labels: List[int],
    gen_probs: List[float],
    gen_labels: List[int],
    thresholds: List[float] | None = None,
    gen_miss_target: float = 0.035,
    gen_fpr_cap: float = 0.15,
    refine: bool = True,
    refine_step: float = 0.002,
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, object]]:
    """Select thresholds that first meet a miss target then minimize FPR.

    Primary filtering keeps thresholds with generalization miss within gen_miss_target.
    If no threshold satisfies the miss target, the selection falls back to the
    low-miss strategy or the minimum miss under the FPR cap and records a warning.
    Within the filtered set the order is: minimize gen FPR, then maximize gen F1,
    then minimize val FPR.
    """

    if thresholds is None:
        thresholds = _generate_threshold_grid()

    val_arr = np.array(val_probs)
    gen_arr = np.array(gen_probs)

    def _eval(thr: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        val_preds = (val_arr >= thr).astype(int).tolist()
        gen_preds = (gen_arr >= thr).astype(int).tolist()
        return confusion_metrics(val_labels, val_preds), confusion_metrics(gen_labels, gen_preds)

    def _select(thr_list: List[float]) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, object]]:
        candidates = []
        best_thr = thr_list[0]
        best_val: Dict[str, float] = {}
        best_gen: Dict[str, float] = {}
        best_record: Dict[str, object] = {}

        for thr in thr_list:
            val_metrics, gen_metrics = _eval(thr)
            in_target = gen_metrics["miss_rate"] <= gen_miss_target
            candidates.append(
                {
                    "threshold": float(thr),
                    "gen_miss": gen_metrics["miss_rate"],
                    "gen_fpr": gen_metrics["fpr"],
                    "gen_f1": gen_metrics["f1"],
                    "val_fpr": val_metrics["fpr"],
                    "val_miss": val_metrics["miss_rate"],
                    "in_target": in_target,
                }
            )

            if not in_target:
                continue

            if best_gen == {}:
                best_thr = float(thr)
                best_val = val_metrics
                best_gen = gen_metrics
                best_record = {
                    "threshold": best_thr,
                    "gen_miss": best_gen["miss_rate"],
                    "gen_fpr": best_gen["fpr"],
                    "gen_f1": best_gen["f1"],
                    "val_fpr": best_val["fpr"],
                    "val_miss": best_val["miss_rate"],
                }
                continue

            better_fpr = gen_metrics["fpr"] < best_gen["fpr"]
            fpr_tie = np.isclose(gen_metrics["fpr"], best_gen["fpr"], atol=1e-8)
            better_f1 = gen_metrics["f1"] > best_gen["f1"]
            f1_tie = np.isclose(gen_metrics["f1"], best_gen["f1"], atol=1e-8)
            better_val_fpr = val_metrics["fpr"] < best_val["fpr"]

            if better_fpr or (fpr_tie and (better_f1 or (f1_tie and better_val_fpr))):
                best_thr = float(thr)
                best_val = val_metrics
                best_gen = gen_metrics
                best_record = {
                    "threshold": best_thr,
                    "gen_miss": best_gen["miss_rate"],
                    "gen_fpr": best_gen["fpr"],
                    "gen_f1": best_gen["f1"],
                    "val_fpr": best_val["fpr"],
                    "val_miss": best_val["miss_rate"],
                }

        info = {
            "gen_miss_target": gen_miss_target,
            "gen_fpr_cap": gen_fpr_cap,
            "candidates": candidates,
            "best_record": best_record,
        }
        return best_thr, best_val, best_gen, info

    best_thr, best_val_metrics, best_gen_metrics, info = _select(thresholds)
    warning = None

    if info.get("best_record", {}) == {}:
        fallback_thr, fallback_val, fallback_gen, fallback_info = sweep_thresholds_low_miss(
            val_probs,
            val_labels,
            gen_probs,
            gen_labels,
            thresholds=thresholds,
            gen_fpr_cap=gen_fpr_cap,
            refine=refine,
            refine_step=refine_step,
        )
        best_thr, best_val_metrics, best_gen_metrics = fallback_thr, fallback_val, fallback_gen
        warning = "no threshold met miss target; fell back to low-miss selection"
        info["warning"] = warning
        info["fallback"] = fallback_info
        info["refined"] = False
        return best_thr, best_val_metrics, best_gen_metrics, info

    if refine:
        window = 0.05
        refine_low = max(0.0, best_thr - window)
        refine_high = min(1.0, best_thr + window)
        fine_grid = np.arange(refine_low, refine_high + 1e-8, refine_step).tolist()
        best_thr, best_val_metrics, best_gen_metrics, info_refined = _select(fine_grid)
        if info_refined.get("best_record", {}) == {}:
            info_refined = info
        info_refined["refined"] = True
        info = info_refined
    else:
        info["refined"] = False

    info["warning"] = warning
    return best_thr, best_val_metrics, best_gen_metrics, info


def sweep_thresholds_three_level(
    val_probs: List[float],
    val_labels: List[int],
    gen_probs: List[float],
    gen_labels: List[int],
    thresholds: List[float] | None = None,
    balanced_miss_cap: float = 0.05,
    balanced_fpr_cap: float = 0.12,
    low_miss_fpr_cap: float = 0.20,
    low_fpr_miss_cap: float = 0.10,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, object]]:
    """Return three threshold tiers under miss/FPR caps for generalization metrics.

    Tiers:
        - high_miss_low_fpr: minimize gen FPR with miss <= low_fpr_miss_cap.
        - balanced: maximize balanced score with miss <= balanced_miss_cap and fpr <= balanced_fpr_cap.
        - low_miss_high_fpr: minimize gen miss with fpr <= low_miss_fpr_cap.
    """

    if thresholds is None:
        thresholds = _generate_threshold_grid()

    val_arr = np.array(val_probs)
    gen_arr = np.array(gen_probs)

    candidates: List[Dict[str, object]] = []
    for thr in thresholds:
        val_preds = (val_arr >= thr).astype(int).tolist()
        gen_preds = (gen_arr >= thr).astype(int).tolist()
        val_metrics = confusion_metrics(val_labels, val_preds)
        gen_metrics = confusion_metrics(gen_labels, gen_preds)
        candidates.append(
            {
                "threshold": float(thr),
                "val_metrics": val_metrics,
                "gen_metrics": gen_metrics,
            }
        )

    balanced_candidates = [
        c
        for c in candidates
        if c["gen_metrics"]["miss_rate"] <= balanced_miss_cap
        and c["gen_metrics"]["fpr"] <= balanced_fpr_cap
    ]
    low_miss_candidates = [c for c in candidates if c["gen_metrics"]["fpr"] <= low_miss_fpr_cap]
    low_fpr_candidates = [c for c in candidates if c["gen_metrics"]["miss_rate"] <= low_fpr_miss_cap]

    warning = None
    if not balanced_candidates:
        warning = "no threshold satisfied balanced miss/fpr caps; using unconstrained candidates for balanced tier"
        balanced_candidates = candidates
    if not low_miss_candidates:
        warning = (
            (warning + " | ") if warning else ""
        ) + "no threshold satisfied low-miss fpr cap; using unconstrained candidates for low-miss tier"
        low_miss_candidates = candidates
    if not low_fpr_candidates:
        warning = (
            (warning + " | ") if warning else ""
        ) + "no threshold satisfied low-fpr miss cap; using unconstrained candidates for low-fpr tier"
        low_fpr_candidates = candidates

    def _pick_high_miss_low_fpr(entries: List[Dict[str, object]]) -> Dict[str, object]:
        return sorted(
            entries,
            key=lambda c: (
                c["gen_metrics"]["fpr"],
                -c["gen_metrics"]["miss_rate"],
                -c["gen_metrics"]["f1"],
                c["val_metrics"]["fpr"],
            ),
        )[0]

    def _pick_balanced(entries: List[Dict[str, object]]) -> Dict[str, object]:
        def _score(c: Dict[str, object]) -> float:
            gen = c["gen_metrics"]
            miss_norm = gen["miss_rate"] / max(balanced_miss_cap, 1e-8)
            fpr_norm = gen["fpr"] / max(balanced_fpr_cap, 1e-8)
            return gen["f1"] - 0.5 * miss_norm - 0.5 * fpr_norm

        return max(
            entries,
            key=lambda c: (
                _score(c),
                c["gen_metrics"]["f1"],
                -c["gen_metrics"]["miss_rate"],
                -c["gen_metrics"]["fpr"],
            ),
        )

    def _pick_low_miss_high_fpr(entries: List[Dict[str, object]]) -> Dict[str, object]:
        return sorted(
            entries,
            key=lambda c: (
                c["gen_metrics"]["miss_rate"],
                -c["gen_metrics"]["fpr"],
                -c["gen_metrics"]["f1"],
                c["val_metrics"]["fpr"],
            ),
        )[0]

    selections = {
        "high_miss_low_fpr": _pick_high_miss_low_fpr(low_fpr_candidates),
        "balanced": _pick_balanced(balanced_candidates),
        "low_miss_high_fpr": _pick_low_miss_high_fpr(low_miss_candidates),
    }

    thresholds_out = {k: float(v["threshold"]) for k, v in selections.items()}
    val_metrics_out = {k: v["val_metrics"] for k, v in selections.items()}
    gen_metrics_out = {k: v["gen_metrics"] for k, v in selections.items()}
    info = {
        "balanced_miss_cap": balanced_miss_cap,
        "balanced_fpr_cap": balanced_fpr_cap,
        "low_miss_fpr_cap": low_miss_fpr_cap,
        "low_fpr_miss_cap": low_fpr_miss_cap,
        "warning": warning,
        "candidates": len(candidates),
        "balanced_candidates": len(balanced_candidates),
        "low_miss_candidates": len(low_miss_candidates),
        "low_fpr_candidates": len(low_fpr_candidates),
    }

    return thresholds_out, val_metrics_out, gen_metrics_out, info


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


class BalancedBatchSampler(Sampler[List[int]]):
    """Yield approximately balanced batches (normal/abnormal) with replacement.

    This sampler cycles minority/majority indices independently to avoid batch-level
    collapse without exhausting either pool. It is designed for binary labels 0/1
    and keeps per-batch composition close to 50/50 when possible.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        minority_label: int = 1,
        majority_label: int = 0,
    ) -> None:
        self.batch_size = batch_size
        self.minority_label = minority_label
        self.majority_label = majority_label

        idx_minor = np.where(labels == minority_label)[0].tolist()
        idx_major = np.where(labels == majority_label)[0].tolist()

        if len(idx_minor) == 0 or len(idx_major) == 0:
            raise ValueError("BalancedBatchSampler requires both classes to be present")

        self.idx_minor = idx_minor
        self.idx_major = idx_major

        self.num_batches = int(np.ceil(len(labels) / batch_size))

    def __iter__(self) -> Iterable[List[int]]:  # type: ignore[override]
        minor_cycle = iter(np.resize(self.idx_minor, self.num_batches * (self.batch_size // 2)))
        major_cycle = iter(np.resize(self.idx_major, self.num_batches * (self.batch_size // 2)))

        for _ in range(self.num_batches):
            batch: List[int] = []
            half = max(1, self.batch_size // 2)
            for _ in range(half):
                batch.append(next(minor_cycle))
                batch.append(next(major_cycle))
            # If batch_size is odd, pad with a majority sample to avoid over-duplicating minority
            if len(batch) < self.batch_size:
                try:
                    batch.append(next(major_cycle))
                except StopIteration:
                    batch.append(self.idx_major[-1])
            yield batch

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_batches
