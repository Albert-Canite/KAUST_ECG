"""Evaluate PBR attenuation with 8-bit quantized inputs and weights."""
from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data import ECGBeatDataset, load_records, set_seed
from models.student import SegmentAwareStudent
from utils import confusion_metrics


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


def quantize_tensor_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits is None or bits < 1:
        return x
    qmax = (2 ** (bits - 1)) - 1
    max_abs = x.abs().max()
    if max_abs < 1e-12 or qmax <= 0:
        return torch.zeros_like(x)
    scale = max_abs / qmax
    return torch.round(x / scale) * scale


def quantize_beats(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits is None or bits < 1:
        return x
    levels = 2 ** bits
    x_clipped = torch.clamp(x, -1.0, 1.0)
    scaled = (x_clipped + 1.0) / 2.0
    quantized = torch.round(scaled * (levels - 1)) / (levels - 1)
    return quantized * 2.0 - 1.0


def quantize_model_weights(model: SegmentAwareStudent, bits: int) -> SegmentAwareStudent:
    quantized = copy.deepcopy(model)
    if bits is None or bits < 1:
        return quantized
    with torch.no_grad():
        for _, param in quantized.named_parameters():
            param.copy_(quantize_tensor_symmetric(param, bits))
    return quantized


def _collect_peak_windows(beat: np.ndarray, baseline: float, peak_window: int, min_prominence: float) -> List[tuple[int, int]]:
    windows: List[tuple[int, int]] = []
    for idx in range(1, len(beat) - 1):
        if beat[idx] > beat[idx - 1] and beat[idx] >= beat[idx + 1] and abs(beat[idx] - baseline) >= min_prominence:
            start = max(0, idx - peak_window)
            end = min(len(beat), idx + peak_window + 1)
            windows.append((start, end))
    if not windows:
        peak_idx = int(np.argmax(np.abs(beat - baseline)))
        start = max(0, peak_idx - peak_window)
        end = min(len(beat), peak_idx + peak_window + 1)
        windows.append((start, end))
    return windows


def attenuate_pbr(
    beat: np.ndarray,
    factor: float,
    peak_window: int = 12,
    min_prominence: float = 0.05,
) -> np.ndarray:
    baseline = float(np.median(beat))
    windows = _collect_peak_windows(beat, baseline, peak_window, min_prominence)
    adjusted = beat.copy()
    for start, end in windows:
        adjusted[start:end] = baseline + factor * (adjusted[start:end] - baseline)
    return adjusted


def apply_pbr_attenuation(
    beats: torch.Tensor,
    factor: float,
    peak_window: int = 12,
    min_prominence: float = 0.05,
) -> torch.Tensor:
    beats_np = beats.squeeze(1).cpu().numpy()
    adjusted = np.stack([attenuate_pbr(b, factor, peak_window, min_prominence) for b in beats_np], axis=0)
    adjusted = torch.from_numpy(adjusted).unsqueeze(1)
    return adjusted


def renormalize_to_unit(beats: torch.Tensor) -> torch.Tensor:
    max_abs = beats.abs().amax(dim=2, keepdim=True)
    safe = torch.where(max_abs > 1e-6, beats / max_abs, beats)
    return torch.clamp(safe, -1.0, 1.0)


def load_student(model_path: str, device: torch.device) -> SegmentAwareStudent:
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model = SegmentAwareStudent(
        num_classes=2,
        num_mlp_layers=int(config.get("num_mlp_layers", 1)),
        dropout_rate=float(config.get("dropout_rate", 0.0)),
        use_value_constraint=bool(config.get("use_value_constraint", True)),
        use_tanh_activations=bool(config.get("use_tanh_activations", False)),
        constraint_scale=float(config.get("constraint_scale", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["student_state_dict"])
    model.eval()
    return model


def evaluate_model(
    model: SegmentAwareStudent,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
    pbr_factor: float,
    input_bits: int,
    peak_window: int,
    min_prominence: float,
    renormalize: bool,
) -> Dict[str, float]:
    preds: List[int] = []
    trues: List[int] = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            adjusted = apply_pbr_attenuation(signals, pbr_factor, peak_window, min_prominence)
            if renormalize:
                adjusted = renormalize_to_unit(adjusted)
            adjusted = quantize_beats(adjusted.to(device), input_bits)
            logits, _ = model(adjusted)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob_pos >= threshold).long()
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    return confusion_metrics(trues, preds)


def plot_rates(pbr_values: List[float], fnr: List[float], fpr: List[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pbr_values, fnr, marker="o", label="FNR")
    ax.plot(pbr_values, fpr, marker="o", label="FPR")
    ax.set_xlabel("PBR Scale")
    ax.set_ylabel("Rate")
    ax.set_title("FNR/FPR vs. PBR Attenuation (8-bit input/weights)")
    ax.set_xticks(pbr_values)
    ax.set_ylim(0.0, max(fnr + fpr) * 1.1 if fnr or fpr else 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pbr_examples(
    beat: np.ndarray,
    output_path: str,
    pbr_values: List[float],
    peak_window: int,
    min_prominence: float,
    renormalize: bool,
    input_bits: int,
) -> None:
    titles = [f"{val:.1f}" for val in pbr_values]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, val in enumerate(pbr_values):
        adjusted = attenuate_pbr(beat, val, peak_window, min_prominence)
        adjusted_tensor = torch.from_numpy(adjusted).reshape(1, 1, -1)
        if renormalize:
            adjusted_tensor = renormalize_to_unit(adjusted_tensor)
        adjusted_tensor = quantize_beats(adjusted_tensor, input_bits)
        adjusted_np = adjusted_tensor.squeeze().cpu().numpy()
        axes[idx].plot(adjusted_np, linewidth=1.0)
        axes[idx].set_title(f"PBR {titles[idx]}")
        axes[idx].set_ylim(0.0, 1.1)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(0.0, 1.1)

    fig.suptitle("Normal Beat with PBR Attenuation")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PBR attenuation sweep with 8-bit quantization")
    parser.add_argument(
        "--data_path",
        type=str,
        default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/",
        help="Path to MIT-BIH dataset root",
    )
    parser.add_argument("--model_path", type=str, default="saved_models/student_model.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.346)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input_bits", type=int, default=8)
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--peak_window", type=int, default=12)
    parser.add_argument("--min_prominence", type=float, default=0.05, help="Minimum absolute prominence for peak detection")
    parser.add_argument(
        "--renormalize_after_pbr",
        action="store_true",
        default=True,
        help="Renormalize each beat to unit amplitude after PBR attenuation (recommended for fair comparison)",
    )
    parser.add_argument("--output", type=str, default="artifacts/quantization_pbr_rates.png")
    parser.add_argument("--example_output", type=str, default="artifacts/quantization_pbr_examples.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_student(args.model_path, device)
    model = quantize_model_weights(model, args.weight_bits)

    beats, labels = load_records(GENERALIZATION_RECORDS, args.data_path)
    if beats.size == 0:
        raise RuntimeError("No beats loaded; check data_path and record availability.")

    dataset = ECGBeatDataset(beats, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    normal_indices = (labels == 0).nonzero()[0]
    if normal_indices.size == 0:
        raise RuntimeError("No normal beats found in gen dataset for PBR example plot.")
    normal_beat = beats[int(normal_indices[0])]
    pbr_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    os.makedirs(os.path.dirname(args.example_output) or ".", exist_ok=True)
    plot_pbr_examples(
        normal_beat,
        args.example_output,
        pbr_values,
        args.peak_window,
        args.min_prominence,
        args.renormalize_after_pbr,
        args.input_bits,
    )
    print(f"Saved PBR example plot to {args.example_output}")

    sweep_values = pbr_values[1:]
    fnr_list: List[float] = []
    fpr_list: List[float] = []
    for val in sweep_values:
        metrics = evaluate_model(
            model,
            loader,
            device,
            args.threshold,
            val,
            args.input_bits,
            args.peak_window,
            args.min_prominence,
            args.renormalize_after_pbr,
        )
        fnr_list.append(metrics["miss_rate"])
        fpr_list.append(metrics["fpr"])
        print(
            f"PBR={val:.1f}: FNR={metrics['miss_rate']:.4f} FPR={metrics['fpr']:.4f} "
            f"(TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']})"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot_rates(sweep_values, fnr_list, fpr_list, args.output)
    print(f"Saved PBR sweep plot to {args.output}")


if __name__ == "__main__":
    main()
