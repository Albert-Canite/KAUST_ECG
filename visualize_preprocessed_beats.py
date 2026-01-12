"""Visualize preprocessed ECG beats from the generalization dataset."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from data import DEFAULT_BEAT_WINDOW, load_records, set_seed
from train_hardware import GENERALIZATION_RECORDS


def _sample_indices(total: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=int)
    replace = total < count
    return rng.choice(total, size=count, replace=replace)


def _plot_group(
    beats: np.ndarray,
    output_path: str,
    rows: int,
    cols: int,
) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 1.8))
    axes = np.atleast_2d(axes)
    for idx, ax in enumerate(axes.flat):
        if idx < len(beats):
            ax.plot(beats[idx], linewidth=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, beats.shape[1] - 1)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/",
        help="Path containing MIT-BIH *.dat/*.hea/*.atr files.",
    )
    parser.add_argument(
        "--output_dir",
        default="test_preprocess",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=5,
        help="Number of grids to generate.",
    )
    parser.add_argument(
        "--beats_per_group",
        type=int,
        default=40,
        help="Number of beats per grid.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows in each grid (10 beats per column).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in each grid.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_BEAT_WINDOW,
        help="Beat window size in samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling beats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    beats, _ = load_records(GENERALIZATION_RECORDS, args.data_path, args.window_size)
    if beats.size == 0:
        raise RuntimeError("No beats loaded from the generalization dataset.")

    total_needed = args.num_groups * args.beats_per_group
    indices = _sample_indices(len(beats), total_needed, rng)
    if len(indices) == 0:
        raise RuntimeError("Unable to sample beats from the dataset.")

    os.makedirs(args.output_dir, exist_ok=True)

    for group_idx in range(args.num_groups):
        start = group_idx * args.beats_per_group
        end = start + args.beats_per_group
        group_indices = indices[start:end]
        group_beats = beats[group_indices]
        output_path = os.path.join(args.output_dir, f"preprocess_group_{group_idx + 1}.png")
        _plot_group(group_beats, output_path, args.rows, args.cols)


if __name__ == "__main__":
    main()
