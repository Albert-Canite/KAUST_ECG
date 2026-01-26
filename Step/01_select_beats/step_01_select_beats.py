"""Step 01: load data, run fixed hardware inference, and select top beats."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ECGBeatDataset, load_records, set_seed
from inference_demo import (
    _collect_processed_outputs,
    apply_checkpoint_hardware_args,
    compute_best_threshold,
    label_names,
    load_student,
    select_top_beats,
)
from train_hardware import GENERALIZATION_RECORDS
from Step.step_utils import copy_output_to_next


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 01: select beats for inference demo.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Step/01_select_beats")
    parser.add_argument("--num_per_class", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input_bits", type=int, default=5)
    parser.add_argument("--weight_bits", type=int, default=5)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--pbr_factor", type=float, default=0.6)
    parser.add_argument("--pbr_peak_window", type=int, default=12)
    parser.add_argument("--pbr_min_prominence", type=float, default=0.05)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--renormalize_inputs", dest="renormalize_inputs", action="store_true", default=True)
    parser.add_argument("--no-renormalize_inputs", dest="renormalize_inputs", action="store_false")
    parser.add_argument("--zero_mean_inputs", dest="zero_mean_inputs", action="store_true", default=True)
    parser.add_argument("--no-zero_mean_inputs", dest="zero_mean_inputs", action="store_false")
    parser.add_argument("--use_checkpoint_hardware", dest="use_checkpoint_hardware", action="store_true", default=True)
    parser.add_argument("--no-use_checkpoint_hardware", dest="use_checkpoint_hardware", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    beats, labels = load_records(GENERALIZATION_RECORDS, args.data_path)
    if beats.size == 0:
        raise RuntimeError("No generalization data loaded. Check data_path.")

    input_path = os.path.join(args.output_dir, "input_01.npz")
    np.savez(input_path, beats=beats, labels=labels)

    model, config = load_student(args.model_path, device)
    if args.use_checkpoint_hardware:
        apply_checkpoint_hardware_args(args, config)

    dataset = ECGBeatDataset(beats, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    labels_arr, probs_arr, processed_beats, logits_arr, pooled_arr = _collect_processed_outputs(
        model,
        loader,
        device,
        args.input_bits,
        args.weight_bits,
        args.snr_db,
        args.pbr_factor,
        args.pbr_peak_window,
        args.pbr_min_prominence,
        args.eval_seed,
        args.renormalize_inputs,
        args.zero_mean_inputs,
    )

    best_threshold = compute_best_threshold(labels_arr, probs_arr)
    _, selected_labels, qualities, selected_preds, selected_indices = select_top_beats(
        beats,
        labels_arr,
        probs_arr,
        best_threshold,
        args.num_per_class,
    )
    label_text = label_names(selected_labels)
    selected_processed = processed_beats[selected_indices].squeeze(1)
    selected_logits = logits_arr[selected_indices]
    selected_probs = probs_arr[selected_indices]
    selected_pooled = pooled_arr[selected_indices]

    output_path = os.path.join(args.output_dir, "output_01.npz")
    np.savez(
        output_path,
        selected_indices=selected_indices,
        selected_labels=selected_labels,
        label_text=np.array(label_text),
        selected_processed=selected_processed,
        selected_logits=selected_logits,
        selected_probs=selected_probs,
        selected_pooled=selected_pooled,
        qualities=np.array(qualities),
        selected_preds=selected_preds,
        best_threshold=best_threshold,
    )
    copy_output_to_next(output_path, "Step/02_segments", "input_02.npz")


if __name__ == "__main__":
    main()
