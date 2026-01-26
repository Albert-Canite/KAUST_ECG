"""Step 04: export MLP1 output."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from inference_demo import load_student, write_mlp_csv
from train_hardware import quantized_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 04: export MLP1 output.")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--weight_bits", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="Step/04_mlp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    data = np.load(args.input_npz, allow_pickle=True)
    tokens_matrix = data["tokens_matrix"]
    label_text = data["label_text"].tolist()

    np.savez(os.path.join(args.output_dir, "input_04_1.npz"), tokens_matrix=tokens_matrix, label_text=label_text)

    model, _ = load_student(args.model_path, device)
    activation = torch.tanh if model.use_tanh_activations else F.relu

    h = torch.from_numpy(tokens_matrix.astype(np.float32)).to(device)

    with quantized_weights(model, bits=args.weight_bits):
        layer = list(model.mlp_layers)[:3][0]
        h = model._scale_if_needed(h)
        h = layer(h)
        h = activation(h)
        weights = layer.weight.detach().cpu().numpy()
        write_mlp_csv(
            "01_mlp_1.csv",
            h,
            args.output_dir,
            label_text,
            weights,
        )

    np.savez(
        os.path.join(args.output_dir, "output_04_1.npz"),
        mlp1_output=h.detach().cpu().numpy(),
        label_text=label_text,
    )


if __name__ == "__main__":
    main()
