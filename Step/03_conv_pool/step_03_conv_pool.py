"""Step 03: export conv/pool kernel CSVs and matrix input."""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from inference_demo import SEGMENT_SLICES, load_student, pool_to_four, write_kernel_csv, write_matrix_input_csv
from train_hardware import quantized_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 03: export conv/pool outputs.")
    parser.add_argument("--input_npz", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--weight_bits", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="Step/03_conv_pool")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    from Step.step_utils import copy_as_input, copy_output_to_next, resolve_input_path

    input_path = resolve_input_path(args.input_npz, "Step/03_conv_pool/input_03.npz")
    data = np.load(input_path, allow_pickle=True)
    selected_processed = data["selected_processed"]
    label_text = data["label_text"].tolist()

    copy_as_input(input_path, args.output_dir, "input_03.npz")

    segment_tensors: Dict[str, torch.Tensor] = {
        name: torch.from_numpy(selected_processed[:, seg].astype(np.float32)).unsqueeze(1).to(device)
        for name, seg in SEGMENT_SLICES.items()
    }

    model, _ = load_student(args.model_path, device)
    activation = torch.tanh if model.use_tanh_activations else F.relu
    conv_layers = {
        "P": model.conv_p,
        "QRS": model.conv_qrs,
        "T": model.conv_t,
        "GLOBAL": model.conv_global,
    }

    pooled_tokens: List[torch.Tensor] = []
    file_index = 1

    with quantized_weights(model, bits=args.weight_bits):
        for segment_name in ("P", "QRS", "T", "GLOBAL"):
            conv_layer = conv_layers[segment_name]
            conv_out = conv_layer(segment_tensors[segment_name])
            activated = activation(conv_out)

            for kernel_idx in range(conv_out.shape[1]):
                kernel_weights = conv_layer.weight.detach().cpu().numpy()[kernel_idx, 0]
                filename = f"{file_index:02d}_{segment_name}.csv"

                conv_values = conv_out[:, kernel_idx, :].detach().cpu().numpy()
                pooled_tensor = pool_to_four(activated[:, kernel_idx : kernel_idx + 1, :]).squeeze(1)
                pooled_values = pooled_tensor.detach().cpu().numpy()

                pooled_tokens.append(pooled_tensor)
                write_kernel_csv(
                    args.output_dir,
                    filename,
                    kernel_weights,
                    pooled_values,
                    conv_values,
                    label_text,
                )
                file_index += 1

        tokens_matrix = torch.stack(pooled_tokens, dim=1)

    write_matrix_input_csv(
        f"{file_index:02d}_matrix_input.csv",
        tokens_matrix,
        args.output_dir,
        label_text,
    )

    output_path = os.path.join(args.output_dir, "output_03.npz")
    np.savez(
        output_path,
        tokens_matrix=tokens_matrix.detach().cpu().numpy(),
        label_text=label_text,
    )
    copy_output_to_next(output_path, "Step/04_mlp", "input_04.npz")


if __name__ == "__main__":
    main()
