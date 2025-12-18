import math
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def build_weighted_sampler(labels: np.ndarray, target_mix=None) -> Tuple[WeightedRandomSampler, Dict[int, int]]:
    if target_mix is None:
        target_mix = {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}
    counts = {i: int((labels == i).sum()) for i in range(4)}
    weights = []
    for lbl in labels:
        lbl_int = int(lbl)
        w = target_mix[lbl_int] / max(counts[lbl_int], 1)
        weights.append(w)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, counts


def effective_num_weights(counts: Dict[int, int], beta: float = 0.9999) -> torch.Tensor:
    weights = []
    for c in range(4):
        n = counts.get(c, 0)
        effective_num = 1.0 - beta ** n
        class_weight = (1.0 - beta) / max(effective_num, 1e-8)
        weights.append(class_weight)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * 4.0
    return weights


def run_sampler_sanity_check(loader: DataLoader, num_batches: int = 20) -> None:
    counts = {i: 0 for i in range(4)}
    weight_samples = []
    sampler = loader.sampler
    if isinstance(sampler, WeightedRandomSampler):
        weight_samples = list(sampler.weights[: min(len(sampler.weights), 20)])
    for idx, (x, y) in enumerate(loader):
        if idx >= num_batches:
            break
        unique, freq = torch.unique(y, return_counts=True)
        for u, f in zip(unique.tolist(), freq.tolist()):
            counts[int(u)] += int(f)
    print(f"[SanityCheck] First {num_batches} batches label counts: {counts}")
    if weight_samples:
        print(f"[SanityCheck] Sample of sampler weights: {weight_samples}")
    if counts[1] == 0 or counts[2] == 0 or counts[3] == 0:
        raise RuntimeError("Sampler sanity check failed: minority classes not sampled in first batches")
