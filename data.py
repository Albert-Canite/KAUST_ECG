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
BEAT_LABEL_MAP = {"N": 0, "V": 1, "S": 1, "F": 1, "Q": 1}


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
        if sym in BEAT_LABEL_MAP:
            s = rpos - window_size // 2
            e = rpos + window_size // 2
            if s >= 0 and e <= len(raw):
                win = raw[s:e].copy()
                win = preprocess_beat(win)
                samples.append(win)
                labels.append(BEAT_LABEL_MAP[sym])
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


def split_dataset(data: np.ndarray, labels: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.permutation(len(data))
    data, labels = data[idx], labels[idx]
    n_train = int((1 - val_ratio) * len(data))
    return data[:n_train], labels[:n_train], data[n_train:], labels[n_train:]


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
