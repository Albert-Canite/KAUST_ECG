"""Step 05: export classifier summary."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from inference_demo import load_student, write_classification_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 05: export classifier summary.")
    parser.add_argument("--input_npz", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output_dir", type=str, default="Step/05_classifier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    from Step.step_utils import copy_as_input, resolve_input_path

    input_path = resolve_input_path(args.input_npz, "Step/05_classifier/input_05.npz")
    data = np.load(input_path, allow_pickle=True)
    selected_logits = data["selected_logits"]
    selected_pooled = data["selected_pooled"]
    label_text = data["label_text"].tolist()

    copy_as_input(input_path, args.output_dir, "input_05.npz")

    model, _ = load_student(args.model_path, device)
    classifier_weights = model.classifier.weight.detach().cpu().numpy()

    logits_tensor = torch.from_numpy(selected_logits).to(device)
    pooled_tensor = torch.from_numpy(selected_pooled).to(device)
    final_probs = torch.softmax(logits_tensor, dim=1)

    write_classification_summary_csv(
        "01_classification.csv",
        pooled_tensor,
        logits_tensor,
        final_probs,
        args.output_dir,
        args.threshold,
        label_text,
        classifier_weights,
    )

    np.savez(
        os.path.join(args.output_dir, "output_05.npz"),
        selected_logits=selected_logits,
        selected_pooled=selected_pooled,
        label_text=label_text,
    )


if __name__ == "__main__":
    main()
