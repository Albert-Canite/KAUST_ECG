"""Compare ROC curves and miss/FPR for two saved models under fixed hardware conditions."""
from __future__ import annotations

import argparse
import os
import re
from contextlib import nullcontext
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import auc

from data import ECGBeatDataset, load_records, set_seed
from train_hardware import (
    GENERALIZATION_RECORDS,
    apply_hardware_effects,
    build_student,
    quantized_weights,
    renormalize_to_unit,
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
    use_hardware: bool,
    use_quantized_weights: bool,
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
                if use_hardware:
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
                else:
                    signals = signals.to(device)
                    if zero_mean_inputs:
                        signals = signals - signals.mean(dim=-1, keepdim=True)
                    if renormalize_inputs:
                        signals = renormalize_to_unit(signals)
                weight_context = (
                    quantized_weights(model, bits=weight_bits)
                    if use_quantized_weights
                    else nullcontext()
                )
                with weight_context:
                    logits, _ = model(signals)
                prob_pos = torch.softmax(logits, dim=1)[:, 1]
                trues.extend(labels.cpu().tolist())
                probs.extend(prob_pos.cpu().tolist())
    return trues, probs


def _sweep_roc(
    trues: List[int],
    probs: List[float],
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    fprs: List[float] = []
    tprs: List[float] = []
    probs_arr = np.array(probs)
    for threshold in thresholds:
        pred = (probs_arr >= threshold).astype(int).tolist()
        metrics = confusion_metrics(trues, pred)
        tprs.append(metrics["sensitivity"])
        fprs.append(metrics["fpr"])
    return np.array(fprs), np.array(tprs)


def _compute_roc_stats(
    trues: List[int],
    probs: List[float],
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, int, float, dict]:
    fpr, tpr = _sweep_roc(trues, probs, thresholds)
    order = np.argsort(fpr)
    roc_auc = auc(fpr[order], tpr[order])
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    pred = (np.array(probs) >= best_threshold).astype(int).tolist()
    metrics = confusion_metrics(trues, pred)
    return fpr, tpr, roc_auc, best_idx, best_threshold, metrics


def _plot_roc(
    output_path: str,
    title: str,
    orig_stats: Tuple[np.ndarray, np.ndarray, float, int, float],
    hw_stats: Tuple[np.ndarray, np.ndarray, float, int, float],
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    import matplotlib.pyplot as plt

    orig_fpr, orig_tpr, orig_auc, orig_best_idx, orig_best_threshold = orig_stats
    hw_fpr, hw_tpr, hw_auc, hw_best_idx, hw_best_threshold = hw_stats
    orig_order = np.argsort(orig_fpr)
    hw_order = np.argsort(hw_fpr)

    plt.figure(figsize=(6, 5))
    plt.plot(orig_fpr[orig_order], orig_tpr[orig_order], label=f"Original AUC={orig_auc:.3f}")
    plt.plot(hw_fpr[hw_order], hw_tpr[hw_order], label=f"Hardware AUC={hw_auc:.3f}")
    plt.scatter(
        [orig_fpr[orig_best_idx]],
        [orig_tpr[orig_best_idx]],
        color="tab:blue",
        s=40,
        zorder=3,
        label=f"Orig best thr={orig_best_threshold:.3f}",
    )
    plt.scatter(
        [hw_fpr[hw_best_idx]],
        [hw_tpr[hw_best_idx]],
        color="tab:orange",
        s=40,
        zorder=3,
        label=f"HW best thr={hw_best_threshold:.3f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


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
        "num_mlp_layers": 3,
        "dropout_rate": 0,
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
    parser.add_argument(
        "--software_output_path", type=str, default=os.path.join("artifacts", "roc_compare_software.png")
    )
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
        use_hardware=True,
        use_quantized_weights=True,
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
        use_hardware=True,
        use_quantized_weights=True,
    )

    thresholds = np.linspace(0.0, 1.0, num=1001)
    orig_fpr, orig_tpr, orig_auc, orig_best_idx, orig_best_threshold, orig_metrics = _compute_roc_stats(
        orig_true, orig_probs, thresholds
    )
    hw_fpr, hw_tpr, hw_auc, hw_best_idx, hw_best_threshold, hw_metrics = _compute_roc_stats(
        hw_true, hw_probs, thresholds
    )

    _plot_roc(
        args.output_path,
        "ROC on Generalization Set (Fixed Hardware)",
        (orig_fpr, orig_tpr, orig_auc, orig_best_idx, orig_best_threshold),
        (hw_fpr, hw_tpr, hw_auc, hw_best_idx, hw_best_threshold),
    )

    print("Fixed hardware conditions:")
    print(f"  SNR={args.snr_db:.2f}dB, PBR={args.pbr_factor:.2f}, seed={args.seed}")
    print(f"Original best threshold={orig_best_threshold:.3f}")
    print(f"Hardware best threshold={hw_best_threshold:.3f}")
    print(
        f"Original: miss={orig_metrics['miss_rate'] * 100:.2f}% "
        f"fpr={orig_metrics['fpr'] * 100:.2f}% auc={orig_auc:.3f}"
    )
    print(
        f"Hardware: miss={hw_metrics['miss_rate'] * 100:.2f}% "
        f"fpr={hw_metrics['fpr'] * 100:.2f}% auc={hw_auc:.3f}"
    )
    print(f"Saved ROC plot to {args.output_path}")

    orig_sw_true, orig_sw_probs = _collect_probs(
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
        use_hardware=False,
        use_quantized_weights=False,
    )
    hw_sw_true, hw_sw_probs = _collect_probs(
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
        use_hardware=False,
        use_quantized_weights=False,
    )

    (
        orig_sw_fpr,
        orig_sw_tpr,
        orig_sw_auc,
        orig_sw_best_idx,
        orig_sw_best_threshold,
        orig_sw_metrics,
    ) = _compute_roc_stats(orig_sw_true, orig_sw_probs, thresholds)
    (
        hw_sw_fpr,
        hw_sw_tpr,
        hw_sw_auc,
        hw_sw_best_idx,
        hw_sw_best_threshold,
        hw_sw_metrics,
    ) = _compute_roc_stats(hw_sw_true, hw_sw_probs, thresholds)

    _plot_roc(
        args.software_output_path,
        "ROC on Generalization Set (Software Ideal)",
        (orig_sw_fpr, orig_sw_tpr, orig_sw_auc, orig_sw_best_idx, orig_sw_best_threshold),
        (hw_sw_fpr, hw_sw_tpr, hw_sw_auc, hw_sw_best_idx, hw_sw_best_threshold),
    )

    print("Software ideal conditions:")
    print(f"Original best threshold={orig_sw_best_threshold:.3f}")
    print(f"Hardware best threshold={hw_sw_best_threshold:.3f}")
    print(
        f"Original: miss={orig_sw_metrics['miss_rate'] * 100:.2f}% "
        f"fpr={orig_sw_metrics['fpr'] * 100:.2f}% auc={orig_sw_auc:.3f}"
    )
    print(
        f"Hardware: miss={hw_sw_metrics['miss_rate'] * 100:.2f}% "
        f"fpr={hw_sw_metrics['fpr'] * 100:.2f}% auc={hw_sw_auc:.3f}"
    )
    print(f"Saved ROC plot to {args.software_output_path}")


if __name__ == "__main__":
    main()
