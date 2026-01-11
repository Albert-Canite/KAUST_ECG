"""Generate matrices and intermediate outputs for hardware inference demo."""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import load_records, set_seed
from models.student import SegmentAwareStudent


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


SEGMENT_SLICES = {
    "P": slice(0, 120),
    "QRS": slice(120, 240),
    "T": slice(240, 360),
    "GLOBAL": slice(0, 360),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inference demo CSVs for hardware deployment")
    parser.add_argument(
        "--data_path",
        type=str,
        default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/",
        help="Path to MIT-BIH dataset root",
    )
    parser.add_argument("--model_path", type=str, default="saved_models/student_model_hardware.pth")
    parser.add_argument("--output_dir", type=str, default="matrice")
    parser.add_argument("--num_per_class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_student(model_path: str, device: torch.device) -> SegmentAwareStudent:
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = SegmentAwareStudent(
        num_classes=2,
        num_mlp_layers=int(config.get("num_mlp_layers", 3)),
        dropout_rate=float(config.get("dropout_rate", 0.0)),
        use_value_constraint=bool(config.get("use_value_constraint", True)),
        use_tanh_activations=bool(config.get("use_tanh_activations", False)),
        constraint_scale=float(config.get("constraint_scale", 1.0)),
        use_bias=bool(config.get("use_bias", True)),
        use_constrained_classifier=bool(config.get("use_constrained_classifier", False)),
    ).to(device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("student_state_dict", "state_dict", "model_state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
    model.load_state_dict(state_dict)
    model.eval()
    return model


def quality_scores(beats: np.ndarray) -> np.ndarray:
    return np.std(beats, axis=1)


def select_top_beats(beats: np.ndarray, labels: np.ndarray, num_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    selected_idx: List[int] = []
    for label in (0, 1):
        class_idx = np.where(labels == label)[0]
        if class_idx.size == 0:
            continue
        scores = quality_scores(beats[class_idx])
        top_k = class_idx[np.argsort(scores)[::-1][:num_per_class]]
        selected_idx.extend(top_k.tolist())
    selected_idx = sorted(selected_idx)
    return beats[selected_idx], labels[selected_idx]


def label_names(labels: np.ndarray) -> List[str]:
    return ["normal" if int(v) == 0 else "abnormal" for v in labels]


def write_segment_csv(output_dir: str, segment_name: str, segment: np.ndarray, labels: np.ndarray) -> None:
    os.makedirs(output_dir, exist_ok=True)
    labels_row = np.array(label_names(labels), dtype=object)
    segment_t = segment.T.astype(object)
    rows = [labels_row] + [segment_t[i] for i in range(segment_t.shape[0])]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"{segment_name}.csv"), header=False, index=False)


def safe_weight_token(value: float) -> str:
    text = f"{value:.4f}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def pool_to_four(values: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool1d(values, 4)


def format_vector(values: np.ndarray) -> str:
    return ",".join(f"{v:.6f}" for v in values.tolist())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beats, labels = load_records(GENERALIZATION_RECORDS, args.data_path)
    if beats.size == 0:
        raise RuntimeError("No generalization data loaded. Check data_path.")

    selected_beats, selected_labels = select_top_beats(beats, labels, args.num_per_class)
    label_text = label_names(selected_labels)

    model = load_student(args.model_path, device)
    if len(model.mlp_layers) < 3:
        raise ValueError("Model must have at least 3 MLP layers for this demo.")

    os.makedirs(args.output_dir, exist_ok=True)

    for segment_name, segment_slice in SEGMENT_SLICES.items():
        segment = selected_beats[:, segment_slice]
        write_segment_csv(args.output_dir, segment_name, segment, selected_labels)

    segment_tensors: Dict[str, torch.Tensor] = {
        name: torch.from_numpy(selected_beats[:, seg].astype(np.float32)).unsqueeze(1).to(device)
        for name, seg in SEGMENT_SLICES.items()
    }

    conv_layers = {
        "P": model.conv_p,
        "QRS": model.conv_qrs,
        "T": model.conv_t,
        "GLOBAL": model.conv_global,
    }
    activation = torch.tanh if model.use_tanh_activations else F.relu

    pooled_tokens: List[torch.Tensor] = []

    for segment_name in ("P", "QRS", "T", "GLOBAL"):
        conv_layer = conv_layers[segment_name]
        conv_out = conv_layer(segment_tensors[segment_name])
        activated = activation(conv_out)

        for kernel_idx in range(conv_out.shape[1]):
            kernel_weights = conv_layer.weight.detach().cpu().numpy()[kernel_idx, 0]
            weight_tokens = "_".join(safe_weight_token(w) for w in kernel_weights)
            filename = f"{segment_name}_kernel{kernel_idx}_{weight_tokens}.csv"

            conv_values = conv_out[:, kernel_idx, :].detach().cpu().numpy()
            pooled_tensor = pool_to_four(activated[:, kernel_idx : kernel_idx + 1, :]).squeeze(1)
            pooled_values = pooled_tensor.detach().cpu().numpy()

            pooled_tokens.append(pooled_tensor)

            conv_strings = [format_vector(row) for row in conv_values]
            pool_strings = [format_vector(row) for row in pooled_values]
            kernel_df = pd.DataFrame({"conv_output": conv_strings, "pool_output": pool_strings})
            kernel_df.to_csv(os.path.join(args.output_dir, filename), index=False)

    tokens_matrix = torch.stack(pooled_tokens, dim=1)

    def write_matrix_csv(name: str, matrix: torch.Tensor) -> None:
        matrix_np = matrix.detach().cpu().numpy()
        flattened = matrix_np.reshape(matrix_np.shape[0], -1).T
        rows = [label_text] + [flattened[i].tolist() for i in range(flattened.shape[0])]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.output_dir, name), header=False, index=False)

    write_matrix_csv("matrix_input.csv", tokens_matrix)

    h = tokens_matrix
    for idx, layer in enumerate(list(model.mlp_layers)[:3], start=1):
        h = model._scale_if_needed(h)
        h = layer(h)
        h = activation(h)
        write_matrix_csv(f"mlp_{idx}.csv", h)

    pooled = h.mean(dim=1)
    logits = model.classifier(pooled)
    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = (probs >= 0.5).long()

    result_df = pd.DataFrame(
        {
            "label": label_text,
            "logit_normal": logits[:, 0].detach().cpu().numpy(),
            "logit_abnormal": logits[:, 1].detach().cpu().numpy(),
            "prob_abnormal": probs.detach().cpu().numpy(),
            "pred_label": ["normal" if int(v) == 0 else "abnormal" for v in preds.cpu().numpy()],
        }
    )
    result_df.to_csv(os.path.join(args.output_dir, "classification.csv"), index=False)


if __name__ == "__main__":
    main()
