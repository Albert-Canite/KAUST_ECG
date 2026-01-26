"""Step 02: export sliding-window segment CSVs."""
from __future__ import annotations

import argparse
import os

import numpy as np

from inference_demo import SEGMENT_SLICES, write_segment_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 02: export segment CSVs.")
    parser.add_argument("--input_npz", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="Step/02_segments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from Step.step_utils import copy_as_input, copy_output_to_next, resolve_input_path

    input_path = resolve_input_path(args.input_npz, "Step/02_segments/input_02.npz")
    data = np.load(input_path, allow_pickle=True)
    selected_processed = data["selected_processed"]
    label_text = data["label_text"].tolist()

    input_snapshot = copy_as_input(input_path, args.output_dir, "input_02.npz")

    file_index = 1
    for segment_name, segment_slice in SEGMENT_SLICES.items():
        segment = selected_processed[:, segment_slice]
        for kernel_idx in range(4):
            filename = f"{file_index:02d}_{segment_name}_k{kernel_idx + 1}.csv"
            write_segment_csv(args.output_dir, filename, segment, label_text)
            file_index += 1

    output_path = os.path.join(args.output_dir, "output_02.npz")
    np.savez(output_path, selected_processed=selected_processed, label_text=label_text)
    copy_output_to_next(output_path, "Step/03_conv_pool", "input_03.npz")


if __name__ == "__main__":
    main()
