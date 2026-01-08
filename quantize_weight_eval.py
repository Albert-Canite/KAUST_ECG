"""Quantize model weights and evaluate FNR/FPR across bit widths."""
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


def quantize_model_weights(model: SegmentAwareStudent, bits: int) -> SegmentAwareStudent:
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in quantized.named_parameters():
            param.copy_(quantize_tensor_symmetric(param, bits))
    return quantized


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
) -> Dict[str, float]:
    preds: List[int] = []
    trues: List[int] = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits, _ = model(signals)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob_pos >= threshold).long()
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    return confusion_metrics(trues, preds)


def plot_rates(bits: List[int], fnr: List[float], fpr: List[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bits, fnr, marker="o", label="FNR")
    ax.plot(bits, fpr, marker="o", label="FPR")
    ax.set_xlabel("Quantization Bits")
    ax.set_ylabel("Rate")
    ax.set_title("FNR/FPR vs. Weight Quantization")
    ax.set_xticks(bits)
    ax.set_ylim(0.0, max(fnr + fpr) * 1.1 if fnr or fpr else 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weight quantization sweep on gen dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/",
        help="Path to MIT-BIH dataset root",
    )
    parser.add_argument("--model_path", type=str, default="saved_models/student_model_hardware.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.346)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="artifacts/quantization_weight_rates.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_student(args.model_path, device)

    beats, labels = load_records(GENERALIZATION_RECORDS, args.data_path)
    if beats.size == 0:
        raise RuntimeError("No beats loaded; check data_path and record availability.")

    dataset = ECGBeatDataset(beats, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    bits_list = list(range(8, 0, -1))
    fnr_list: List[float] = []
    fpr_list: List[float] = []
    for bits in bits_list:
        quantized_model = quantize_model_weights(model, bits)
        metrics = evaluate_model(quantized_model, loader, device, args.threshold)
        fnr_list.append(metrics["miss_rate"])
        fpr_list.append(metrics["fpr"])
        print(
            f"bits={bits}: FNR={metrics['miss_rate']:.4f} FPR={metrics['fpr']:.4f} "
            f"(TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']})"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot_rates(bits_list, fnr_list, fpr_list, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
