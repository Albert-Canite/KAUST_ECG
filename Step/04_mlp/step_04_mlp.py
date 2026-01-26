"""Step 04: export MLP layer outputs."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from inference_demo import load_student, write_mlp_csv
from train_hardware import quantized_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 04: export MLP layer outputs.")
    parser.add_argument("--input_npz", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--weight_bits", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="Step/04_mlp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    from Step.step_utils import copy_as_input, copy_output_to_next, resolve_input_path

    input_path = resolve_input_path(args.input_npz, "Step/04_mlp/input_04.npz")
    data = np.load(input_path, allow_pickle=True)
    tokens_matrix = data["tokens_matrix"]
    label_text = data["label_text"].tolist()

    copy_as_input(input_path, args.output_dir, "input_04.npz")

    model, _ = load_student(args.model_path, device)
    activation = torch.tanh if model.use_tanh_activations else F.relu

    h = torch.from_numpy(tokens_matrix.astype(np.float32)).to(device)
    file_index = 1

    with quantized_weights(model, bits=args.weight_bits):
        for idx, layer in enumerate(list(model.mlp_layers)[:3], start=1):
            h = model._scale_if_needed(h)
            h = layer(h)
            h = activation(h)
            weights = layer.weight.detach().cpu().numpy()
            write_mlp_csv(
                f"{file_index:02d}_mlp_{idx}.csv",
                h,
                args.output_dir,
                label_text,
                weights,
            )
            file_index += 1

    output_path = os.path.join(args.output_dir, "output_04.npz")
    np.savez(
        output_path,
        mlp_output=h.detach().cpu().numpy(),
        label_text=label_text,
    )
    copy_output_to_next(output_path, "Step/05_classifier", "input_05.npz")


if __name__ == "__main__":
    main()
