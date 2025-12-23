"""Evaluate quantized model with Gaussian noise injected into beats."""
from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List

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
    if bits < 1:
        raise ValueError("bits must be >= 1")
    qmax = (2 ** (bits - 1)) - 1
    max_abs = x.abs().max()
    if max_abs < 1e-12 or qmax <= 0:
        return torch.zeros_like(x)
    scale = max_abs / qmax
    return torch.round(x / scale) * scale


def quantize_beats(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 1:
        raise ValueError("bits must be >= 1")
    levels = 2 ** bits
    x_clipped = torch.clamp(x, -1.0, 1.0)
    scaled = (x_clipped + 1.0) / 2.0
    quantized = torch.round(scaled * (levels - 1)) / (levels - 1)
    return quantized * 2.0 - 1.0


def quantize_model_weights(model: SegmentAwareStudent, bits: int) -> SegmentAwareStudent:
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for _, param in quantized.named_parameters():
            param.copy_(quantize_tensor_symmetric(param, bits))
    return quantized


def add_gaussian_noise_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    power = x.pow(2).mean(dim=-1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = power / snr_linear
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(x) * noise_std
    return x + noise


def plot_noise_examples(
    beat: torch.Tensor,
    output_path: str,
    snr_values: List[float],
) -> None:
    titles = ["Clean"] + [f"{snr:.0f} dB" for snr in snr_values]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    axes[0].plot(beat.squeeze().cpu().numpy(), linewidth=1.0)
    axes[0].set_title(titles[0])

    for idx, snr in enumerate(snr_values, start=1):
        noisy = add_gaussian_noise_snr(beat, snr)
        axes[idx].plot(noisy.squeeze().cpu().numpy(), linewidth=1.0)
        axes[idx].set_title(titles[idx])

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Normal Beat with Gaussian Noise")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


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
    snr_db: float,
    input_bits: int,
) -> Dict[str, float]:
    preds: List[int] = []
    trues: List[int] = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            noisy = add_gaussian_noise_snr(signals, snr_db)
            noisy = quantize_beats(noisy, input_bits)
            logits, _ = model(noisy)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob_pos >= threshold).long()
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    return confusion_metrics(trues, preds)


def plot_rates(snr_list: List[float], fnr: List[float], fpr: List[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snr_list, fnr, marker="o", label="FNR")
    ax.plot(snr_list, fpr, marker="o", label="FPR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Rate")
    ax.set_title("FNR/FPR vs. Noise Level (8-bit input/weights)")
    ax.set_xticks(snr_list)
    ax.set_ylim(0.0, max(fnr + fpr) * 1.1 if fnr or fpr else 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noise injection sweep with 8-bit quantized model")
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
    parser.add_argument("--snr_min", type=float, default=-10.0)
    parser.add_argument("--snr_max", type=float, default=40.0)
    parser.add_argument("--snr_step", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="artifacts/quantization_noise_rates.png")
    parser.add_argument("--example_output", type=str, default="artifacts/quantization_noise_examples.png")
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
        raise RuntimeError("No normal beats found in gen dataset for noise example plot.")
    normal_beat = torch.from_numpy(beats[int(normal_indices[0])]).reshape(1, 1, -1)
    example_snrs = [40.0, 30.0, 20.0, 10.0, 0.0]
    os.makedirs(os.path.dirname(args.example_output) or ".", exist_ok=True)
    plot_noise_examples(normal_beat, args.example_output, example_snrs)
    print(f"Saved noise example plot to {args.example_output}")

    snr_list = []
    fnr_list: List[float] = []
    fpr_list: List[float] = []
    snr = args.snr_min
    while snr <= args.snr_max + 1e-6:
        snr_list.append(round(snr, 3))
        metrics = evaluate_model(model, loader, device, args.threshold, snr, args.input_bits)
        fnr_list.append(metrics["miss_rate"])
        fpr_list.append(metrics["fpr"])
        print(
            f"SNR={snr:.1f}dB: FNR={metrics['miss_rate']:.4f} FPR={metrics['fpr']:.4f} "
            f"(TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']})"
        )
        snr += args.snr_step

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot_rates(snr_list, fnr_list, fpr_list, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
