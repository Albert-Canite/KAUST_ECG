import os
import random
from typing import Dict, List, Tuple

import numpy as np
import wfdb

TRAIN_RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "115",
    "117",
    "121",
    "122",
    "123",
    "200",
    "201",
    "202",
    "203",
    "207",
    "209",
    "210",
    "212",
    "213",
    "219",
    "222",
    "223",
    "230",
    "231",
    "232",
]

GENERALIZATION_RECORDS = [
    "106",
    "114",
    "116",
    "118",
    "119",
    "124",
    "205",
    "208",
    "214",
    "215",
    "220",
    "221",
    "228",
    "233",
    "234",
]

# AAMI four-class mapping (symbols taken from MIT-BIH annotations)
AAMI_MAP = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "O": ["F", "/", "f", "Q", "?", "[", "]", "|", "~"],
}

SYMBOL_TO_CLASS = {
    symbol: 0 for symbol in AAMI_MAP["N"]
}
SYMBOL_TO_CLASS.update({symbol: 1 for symbol in AAMI_MAP["S"]})
SYMBOL_TO_CLASS.update({symbol: 2 for symbol in AAMI_MAP["V"]})
SYMBOL_TO_CLASS.update({symbol: 3 for symbol in AAMI_MAP["O"]})


class BeatDataset:
    def __init__(
        self,
        data_root: str,
        records: List[str],
        normalization: str = "zscore",
    ) -> None:
        self.data_root = data_root
        self.records = records
        self.normalization = normalization
        self.X, self.y = self._load()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.X[idx], int(self.y[idx])

    def _normalize(self, beat: np.ndarray) -> np.ndarray:
        if self.normalization == "robust":
            median = np.median(beat)
            iqr = np.subtract(*np.percentile(beat, [75, 25]))
            scale = iqr if iqr > 1e-6 else 1.0
            return ((beat - median) / scale).astype(np.float32)
        mean = beat.mean()
        std = beat.std()
        std = std if std > 1e-6 else 1.0
        return ((beat - mean) / std).astype(np.float32)

    def _load_record(self, record: str) -> Tuple[List[np.ndarray], List[int]]:
        record_path = os.path.join(self.data_root, record)
        try:
            rec = wfdb.rdrecord(record_path)
        except FileNotFoundError:
            rec = wfdb.rdrecord(record)
        ch_name_to_idx = {name: i for i, name in enumerate(rec.sig_name)}
        if "MLII" in ch_name_to_idx:
            ch_idx = ch_name_to_idx["MLII"]
            lead_used = "MLII"
        else:
            ch_idx = 0
            lead_used = rec.sig_name[0]
        signal = rec.p_signal[:, ch_idx]
        try:
            ann = wfdb.rdann(record_path, "atr")
        except FileNotFoundError:
            ann = wfdb.rdann(record, "atr")
        beats = []
        labels = []
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol not in SYMBOL_TO_CLASS:
                continue
            start = sample - 180
            end = sample + 180
            if start < 0 or end > len(signal):
                continue
            beat = signal[start:end]
            beat = self._normalize(beat)
            beats.append(beat)
            labels.append(SYMBOL_TO_CLASS[symbol])
        if not beats:
            print(f"[WARN] No beats kept for record {record}")
        else:
            print(f"Loaded {len(beats)} beats from {record} using lead {lead_used}")
        return beats, labels

    def _load(self) -> Tuple[np.ndarray, np.ndarray]:
        all_beats: List[np.ndarray] = []
        all_labels: List[int] = []
        for rec in self.records:
            beats, labels = self._load_record(rec)
            all_beats.extend(beats)
            all_labels.extend(labels)
        if len(all_beats) == 0:
            raise RuntimeError("No beats loaded; please check data_root and record availability")
        X = np.stack(all_beats).astype(np.float32)
        y = np.array(all_labels, dtype=np.int64)
        return X, y


def split_train_val_records(records: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    records = records.copy()
    rng.shuffle(records)
    val_size = max(1, int(len(records) * val_ratio))
    val_records = records[:val_size]
    train_records = records[val_size:]
    return train_records, val_records


def load_datasets(data_root: str, val_ratio: float, seed: int, normalization: str = "zscore"):
    train_records, val_records = split_train_val_records(TRAIN_RECORDS, val_ratio, seed)
    print(f"Train records: {train_records}\nVal records: {val_records}\nGeneralization records: {GENERALIZATION_RECORDS}")
    train_ds = BeatDataset(data_root, train_records, normalization)
    val_ds = BeatDataset(data_root, val_records, normalization)
    gen_ds = BeatDataset(data_root, GENERALIZATION_RECORDS, normalization)
    def count_labels(ds: BeatDataset) -> Dict[int, int]:
        counts = {i: 0 for i in range(4)}
        for lbl in ds.y:
            counts[int(lbl)] += 1
        return counts
    print(f"Train label counts: {count_labels(train_ds)}")
    print(f"Val label counts: {count_labels(val_ds)}")
    print(f"Gen label counts: {count_labels(gen_ds)}")
    return train_ds, val_ds, gen_ds
