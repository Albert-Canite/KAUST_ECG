"""Compare ROC curves and miss/FPR for two saved models under fixed hardware conditions."""
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from data import ECGBeatDataset, load_records, set_seed
from train_hardware import (
    GENERALIZATION_RECORDS,
    apply_hardware_effects,
    build_student,
    quantized_weights,
)
from utils import confusion_metrics


def _collect_probs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    input_bits: int,
    weight_bits: int,
    snr_db: float,
    pbr_factor: float,
    pbr_peak_window: int,
    pbr_min_prominence: float,
    seed: int,
    renormalize_inputs: bool,
    zero_mean_inputs: bool,
) -> Tuple[List[int], List[float]]:
    model.eval()
    trues: List[int] = []
    probs: List[float] = []
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
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
                trues.extend(labels.cpu().tolist())
                probs.extend(prob_pos.cpu().tolist())
    return trues, probs


def _infer_model_overrides(state_dict: dict) -> dict:
    overrides: dict[str, object] = {}
    if not isinstance(state_dict, dict):
        return overrides

    keys = list(state_dict.keys())
    if "classifier.weight_param" in keys:
        overrides["use_constrained_classifier"] = True
    elif "classifier.weight" in keys:
        overrides["use_constrained_classifier"] = False

    if any(key.endswith("weight_param") for key in keys):
        overrides["use_value_constraint"] = True
    elif any(key.endswith(".weight") for key in keys):
        overrides["use_value_constraint"] = False

    if any(key.endswith("bias_param") or key.endswith(".bias") for key in keys):
        overrides["use_bias"] = True

    mlp_indices = set()
    for key in keys:
        match = re.match(r"mlp_layers\.(\d+)\.", key)
        if match:
            mlp_indices.add(int(match.group(1)))
    if mlp_indices:
        overrides["num_mlp_layers"] = max(mlp_indices) + 1

    return overrides


def _build_model_args(config: dict | None, state_dict: dict | None) -> argparse.Namespace:
    defaults = {
        "num_mlp_layers": 1,
        "dropout_rate": 0.1,
        "use_value_constraint": True,
        "use_tanh_activations": False,
        "constraint_scale": 1.0,
        "use_bias": True,
        "use_constrained_classifier": False,
    }
    if config:
        for key in defaults:
            if key in config:
                defaults[key] = config[key]
    if state_dict is not None:
        overrides = _infer_model_overrides(state_dict)
        for key, value in overrides.items():
            if config is None or key not in config:
                defaults[key] = value
    return argparse.Namespace(**defaults)


def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    config = state.get("config") if isinstance(state, dict) else None
    state_dict = state
    if isinstance(state, dict):
        for key in ("state_dict", "student_state_dict", "model_state_dict"):
            if key in state:
                state_dict = state[key]
                break
    model = build_student(_build_model_args(config, state_dict), device)
    model.load_state_dict(state_dict)
    return model


DEFAULT_DATA_PATH = "E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/"
DEFAULT_ORIGINAL_MODEL = os.path.join("saved_models", "student_model.pth")
DEFAULT_HARDWARE_MODEL = os.path.join("saved_models", "student_model_hardware.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ROC curves under fixed hardware conditions")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--original_model", type=str, default=DEFAULT_ORIGINAL_MODEL)
    parser.add_argument("--hardware_model", type=str, default=DEFAULT_HARDWARE_MODEL)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--input_bits", type=int, default=5)
    parser.add_argument("--weight_bits", type=int, default=5)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--pbr_factor", type=float, default=0.6)
    parser.add_argument("--pbr_peak_window", type=int, default=12)
    parser.add_argument("--pbr_min_prominence", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_path", type=str, default=os.path.join("artifacts", "roc_compare.png"))
    parser.add_argument("--renormalize_inputs", dest="renormalize_inputs", action="store_true", default=True)
    parser.add_argument("--no-renormalize_inputs", dest="renormalize_inputs", action="store_false")
    parser.add_argument("--zero_mean_inputs", dest="zero_mean_inputs", action="store_true", default=True)
    parser.add_argument("--no-zero_mean_inputs", dest="zero_mean_inputs", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for label, path in (("original_model", args.original_model), ("hardware_model", args.hardware_model)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} checkpoint not found: {path}. Update DEFAULT_* paths in compare_hardware_roc.py or "
                "pass --original_model/--hardware_model."
            )
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"data_path not found: {args.data_path}. Update DEFAULT_DATA_PATH in compare_hardware_roc.py or "
            "pass --data_path."
        )

    gen_x, gen_y = load_records(GENERALIZATION_RECORDS, args.data_path)
    if gen_x.size == 0:
        raise RuntimeError("No generalization data loaded. Check data_path.")
    gen_loader = torch.utils.data.DataLoader(
        ECGBeatDataset(gen_x, gen_y), batch_size=args.batch_size, shuffle=False
    )

    original_model = _load_model(args.original_model, device)
    hardware_model = _load_model(args.hardware_model, device)

    orig_true, orig_probs = _collect_probs(
        original_model,
        gen_loader,
        device,
        args.input_bits,
        args.weight_bits,
        args.snr_db,
        args.pbr_factor,
        args.pbr_peak_window,
        args.pbr_min_prominence,
        args.seed,
        args.renormalize_inputs,
        args.zero_mean_inputs,
    )
    hw_true, hw_probs = _collect_probs(
        hardware_model,
        gen_loader,
        device,
        args.input_bits,
        args.weight_bits,
        args.snr_db,
        args.pbr_factor,
        args.pbr_peak_window,
        args.pbr_min_prominence,
        args.seed,
        args.renormalize_inputs,
        args.zero_mean_inputs,
    )

    orig_fpr, orig_tpr, _ = roc_curve(orig_true, orig_probs)
    hw_fpr, hw_tpr, _ = roc_curve(hw_true, hw_probs)
    orig_auc = auc(orig_fpr, orig_tpr)
    hw_auc = auc(hw_fpr, hw_tpr)

    orig_pred = (np.array(orig_probs) >= args.threshold).astype(int).tolist()
    hw_pred = (np.array(hw_probs) >= args.threshold).astype(int).tolist()
    orig_metrics = confusion_metrics(orig_true, orig_pred)
    hw_metrics = confusion_metrics(hw_true, hw_pred)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.plot(orig_fpr, orig_tpr, label=f"Original AUC={orig_auc:.3f}")
    plt.plot(hw_fpr, hw_tpr, label=f"Hardware AUC={hw_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC on Generalization Set (Fixed Hardware)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)

    print("Fixed hardware conditions:")
    print(f"  SNR={args.snr_db:.2f}dB, PBR={args.pbr_factor:.2f}, seed={args.seed}")
    print(f"Threshold={args.threshold:.3f}")
    print(
        f"Original: miss={orig_metrics['miss_rate'] * 100:.2f}% "
        f"fpr={orig_metrics['fpr'] * 100:.2f}% auc={orig_auc:.3f}"
    )
    print(
        f"Hardware: miss={hw_metrics['miss_rate'] * 100:.2f}% "
        f"fpr={hw_metrics['fpr'] * 100:.2f}% auc={hw_auc:.3f}"
    )
    print(f"Saved ROC plot to {args.output_path}")


if __name__ == "__main__":
    main()
