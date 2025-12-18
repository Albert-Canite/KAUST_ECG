"""MIT-BIH data utilities."""
from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

MITBIH_SAMPLING_RATE = 360
DEFAULT_BEAT_WINDOW = 360

# Map raw MIT-BIH annotations to 4-class labels: N(0), S(1), V(2), O(3)
# Any annotation not listed explicitly will be treated as class O (others).
BEAT_LABEL_MAP = {
    # Normal-like beats
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    # Supraventricular ectopic beats
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    # Ventricular ectopic beats
    "V": 2,
    "E": 2,
    # Other / unclassified beats
    "F": 3,
    "/": 3,
    "f": 3,
    "Q": 3,
    "U": 3,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_beat(window_1d: np.ndarray, clip_value: float = 1.0) -> np.ndarray:
    w = window_1d.astype(np.float32)
    w = w - np.mean(w)
    max_abs = np.max(np.abs(w))
    if max_abs < 1e-6:
        return np.zeros_like(w, dtype=np.float32)
    w = w / max_abs
    w = np.clip(w, -clip_value, clip_value)
    return w.astype(np.float32)


def load_record(record_id: str, data_path: str, window_size: int = DEFAULT_BEAT_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    record = wfdb.rdrecord(os.path.join(data_path, record_id))
    ann = wfdb.rdann(os.path.join(data_path, record_id), "atr")
    raw = record.p_signal[:, 0].astype(np.float32)

    samples: List[np.ndarray] = []
    labels: List[int] = []
    for sym, rpos in zip(ann.symbol, ann.sample):
        # Default to class O (3) for any annotation not explicitly mapped above.
        label = BEAT_LABEL_MAP.get(sym, 3)
        s = rpos - window_size // 2
        e = rpos + window_size // 2
        if s >= 0 and e <= len(raw):
            win = raw[s:e].copy()
            win = preprocess_beat(win)
            samples.append(win)
            labels.append(label)
    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_records(record_ids: List[str], data_path: str, window_size: int = DEFAULT_BEAT_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for rid in record_ids:
        s, l = load_record(rid, data_path, window_size)
        if s.size and l.size:
            xs.append(s)
            ys.append(l)
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def split_dataset(
    data: np.ndarray, labels: np.ndarray, val_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split to keep every class in the training fold.

    A random shuffle alone can drop ultra-rare classes (notably S-beats) from the
    training split, making the model incapable of predicting them. We instead
    split per-class, keeping at least one sample of any class with >1 examples in
    the training set and placing up to ``val_ratio`` in the validation fold.
    """

    train_indices: List[int] = []
    val_indices: List[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        if len(cls_idx) <= 1:
            # With a single example, keep it for training to avoid zero-shot learning.
            val_count = 0
        else:
            val_count = int(np.floor(val_ratio * len(cls_idx)))
            val_count = max(1, min(val_count, len(cls_idx) - 1))

        val_indices.extend(cls_idx[:val_count].tolist())
        train_indices.extend(cls_idx[val_count:].tolist())

    # Final shuffle so batches remain mixed across classes.
    train_indices = np.random.permutation(train_indices)
    val_indices = np.random.permutation(val_indices)

    return data[train_indices], labels[train_indices], data[val_indices], labels[val_indices]


class ECGBeatDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.data[idx].reshape(1, -1).astype(np.float32)
        y = int(self.labels[idx])
        return torch.from_numpy(x), y
