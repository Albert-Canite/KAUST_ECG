"""Step 02: export P segment (kernel 3 placeholder)."""
from __future__ import annotations

import argparse
import os

import numpy as np

from inference_demo import write_segment_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 02: export P segment for kernel 3.")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Step/02_segments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = np.load(args.input_npz, allow_pickle=True)
    selected_processed = data["selected_processed"]
    label_text = data["label_text"].tolist()
    segment = selected_processed[:, 0:120]
    write_segment_csv(args.output_dir, "03_P_k3.csv", segment, label_text)


if __name__ == "__main__":
    main()
