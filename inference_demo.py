"""Generate matrices and intermediate outputs for a hardware inference demo."""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ECGBeatDataset, load_records, set_seed
from models.student import SegmentAwareStudent
from train_hardware import (
    GENERALIZATION_RECORDS,
    apply_hardware_effects,
    quantized_weights,
)

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


def load_student(model_path: str, device: torch.device) -> Tuple[SegmentAwareStudent, Dict[str, object]]:
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
    return model, config


def apply_checkpoint_hardware_args(args: argparse.Namespace, config: Dict[str, object]) -> None:
    if not config:
        return
    args.input_bits = int(config.get("input_bits", args.input_bits))
    args.weight_bits = int(config.get("weight_bits", args.weight_bits))
    args.renormalize_inputs = bool(config.get("renormalize_inputs", args.renormalize_inputs))
    args.zero_mean_inputs = bool(config.get("zero_mean_inputs", args.zero_mean_inputs))
    args.pbr_peak_window = int(config.get("pbr_peak_window", args.pbr_peak_window))
    args.pbr_min_prominence = float(config.get("pbr_min_prominence", args.pbr_min_prominence))


def apply_fixed_hardware_effects(
    signals: torch.Tensor,
    device: torch.device,
    input_bits: int,
    snr_db: float,
    pbr_factor: float,
    pbr_peak_window: int,
    pbr_min_prominence: float,
    renormalize_inputs: bool,
    zero_mean_inputs: bool,
    eval_seed: int | None,
) -> torch.Tensor:
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices, enabled=eval_seed is not None):
        if eval_seed is not None:
            torch.manual_seed(eval_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed)
        signals = apply_hardware_effects(
            signals,
            device,
            input_bits,
            snr_db,
            snr_db,
            pbr_factor,
            pbr_factor,
            pbr_peak_window,
            pbr_min_prominence,
            hardware_prob=1.0,
            training=False,
            renormalize=renormalize_inputs,
            zero_mean=zero_mean_inputs,
        )
    return signals


def _collect_probs(
    model: SegmentAwareStudent,
    loader: DataLoader,
    device: torch.device,
    input_bits: int,
    weight_bits: int,
    snr_db: float,
    pbr_factor: float,
    pbr_peak_window: int,
    pbr_min_prominence: float,
    eval_seed: int | None,
    renormalize_inputs: bool,
    zero_mean_inputs: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: List[int] = []
    probs_all: List[float] = []
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices, enabled=eval_seed is not None):
        if eval_seed is not None:
            torch.manual_seed(eval_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed)
        with torch.no_grad():
            for signals, labels in loader:
                labels = labels.to(device)
                signals = apply_hardware_effects(
                    signals,
                    device,
                    input_bits,
                    snr_db,
                    snr_db,
                    pbr_factor,
                    pbr_factor,
                    pbr_peak_window,
                    pbr_min_prominence,
                    hardware_prob=1.0,
                    training=False,
                    renormalize=renormalize_inputs,
                    zero_mean=zero_mean_inputs,
                )
                with quantized_weights(model, bits=weight_bits):
                    logits, _ = model(signals)
                prob_pos = torch.softmax(logits, dim=1)[:, 1]
                labels_all.extend(labels.cpu().tolist())
                probs_all.extend(prob_pos.cpu().tolist())
    return np.array(labels_all), np.array(probs_all)


def compute_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.linspace(0.0, 1.0, num=1001)
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tp = int(((labels == 1) & (preds == 1)).sum())
        tn = int(((labels == 0) & (preds == 0)).sum())
        fp = int(((labels == 0) & (preds == 1)).sum())
        fn = int(((labels == 1) & (preds == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def select_top_beats(
    beats: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    num_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = (probs >= threshold).astype(int)
    selected_indices: List[int] = []
    qualities: List[float] = []

    for class_label in (0, 1):
        class_mask = (labels == class_label) & (preds == class_label)
        class_indices = np.where(class_mask)[0]
        if class_indices.size == 0:
            continue
        class_probs = probs[class_indices]
        class_quality = class_probs if class_label == 1 else 1.0 - class_probs
        order = np.argsort(class_quality)[::-1]
        top_indices = class_indices[order][:num_per_class]
        selected_indices.extend(top_indices.tolist())
        qualities.extend(class_quality[order][:num_per_class].tolist())

    if len(selected_indices) < num_per_class * 2:
        raise RuntimeError("Not enough correctly classified beats to cover both classes.")

    selected_indices_arr = np.array(selected_indices)
    qualities_arr = np.array(qualities)
    return beats[selected_indices_arr], labels[selected_indices_arr], qualities_arr


def label_names(labels: Iterable[int]) -> List[str]:
    return ["normal" if int(v) == 0 else "abnormal" for v in labels]


def write_segment_csv(output_dir: str, segment_name: str, segment: np.ndarray, labels: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{segment_name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(labels)
        for row in segment.T:
            writer.writerow(row.tolist())


def safe_weight_token(value: float) -> str:
    text = f"{value:.4f}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def pool_to_four(values: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool1d(values, 4)


def format_vector(values: np.ndarray) -> str:
    return ",".join(f"{v:.6f}" for v in values.tolist())


def write_matrix_csv(name: str, matrix: torch.Tensor, output_dir: str, labels: List[str]) -> None:
    matrix_np = matrix.detach().cpu().numpy()
    flattened = matrix_np.reshape(matrix_np.shape[0], -1).T
    path = os.path.join(output_dir, name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(labels)
        for row in flattened:
            writer.writerow(row.tolist())


def run_demo() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beats, labels = load_records(GENERALIZATION_RECORDS, args.data_path)
    if beats.size == 0:
        raise RuntimeError("No generalization data loaded. Check data_path.")

    model, config = load_student(args.model_path, device)
    if len(model.mlp_layers) < 3:
        raise ValueError("Model must have at least 3 MLP layers for this demo.")

    if args.use_checkpoint_hardware:
        apply_checkpoint_hardware_args(args, config)

    dataset = ECGBeatDataset(beats, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    labels_arr, probs_arr = _collect_probs(
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
    print(f"Best threshold={best_threshold:.3f}")

    selected_beats, selected_labels, qualities = select_top_beats(
        beats,
        labels_arr,
        probs_arr,
        best_threshold,
        args.num_per_class,
    )
    label_text = label_names(selected_labels)

    signals = torch.from_numpy(selected_beats.astype(np.float32)).unsqueeze(1).to(device)
    signals = apply_fixed_hardware_effects(
        signals,
        device,
        args.input_bits,
        args.snr_db,
        args.pbr_factor,
        args.pbr_peak_window,
        args.pbr_min_prominence,
        args.renormalize_inputs,
        args.zero_mean_inputs,
        args.eval_seed,
    )
    processed_beats = signals.squeeze(1).detach().cpu().numpy()

    os.makedirs(args.output_dir, exist_ok=True)

    for segment_name, segment_slice in SEGMENT_SLICES.items():
        segment = processed_beats[:, segment_slice]
        write_segment_csv(args.output_dir, segment_name, segment, label_text)

    segment_tensors: Dict[str, torch.Tensor] = {
        name: torch.from_numpy(processed_beats[:, seg].astype(np.float32)).unsqueeze(1).to(device)
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

    with quantized_weights(model, bits=args.weight_bits):
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

    write_matrix_csv("matrix_input.csv", tokens_matrix, args.output_dir, label_text)

    h = tokens_matrix
    for idx, layer in enumerate(list(model.mlp_layers)[:3], start=1):
        h = model._scale_if_needed(h)
        h = layer(h)
        h = activation(h)
        write_matrix_csv(f"mlp_{idx}.csv", h, args.output_dir, label_text)

    pooled = h.mean(dim=1)
    logits = model.classifier(pooled)
    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = (probs >= best_threshold).long()

    result_df = pd.DataFrame(
        {
            "label": label_text,
            "quality": qualities,
            "logit_normal": logits[:, 0].detach().cpu().numpy(),
            "logit_abnormal": logits[:, 1].detach().cpu().numpy(),
            "prob_abnormal": probs.detach().cpu().numpy(),
            "pred_label": ["normal" if int(v) == 0 else "abnormal" for v in preds.cpu().numpy()],
            "threshold": best_threshold,
        }
    )
    result_df.to_csv(os.path.join(args.output_dir, "classification.csv"), index=False)


if __name__ == "__main__":
    run_demo()
