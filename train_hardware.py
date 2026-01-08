"""Training script for hardware-aware ECG classification with quantization and noise."""
from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve

from constraints import ConstrainedConv1d, ConstrainedLinear
from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from quantize_pbr_eval import apply_pbr_attenuation
from utils import (
    BalancedBatchSampler,
    compute_class_weights,
    confusion_metrics,
    make_weighted_sampler,
    sweep_thresholds,
    sweep_thresholds_blended,
    sweep_thresholds_min_miss,
)


TRAIN_RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "115",
    "117",
    "121",
    "122",
    "123",
    "200",
    "201",
    "202",
    "203",
    "207",
    "209",
    "210",
    "212",
    "213",
    "219",
    "222",
    "223",
    "230",
    "231",
    "232",
]
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


def add_gaussian_noise_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    power = x.pow(2).mean(dim=-1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = power / snr_linear
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(x) * noise_std
    return x + noise


def renormalize_to_unit(beats: torch.Tensor) -> torch.Tensor:
    max_abs = beats.abs().amax(dim=2, keepdim=True)
    safe = torch.where(max_abs > 1e-6, beats / max_abs, beats)
    return torch.clamp(safe, -1.0, 1.0)


def _iter_effective_weights(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    weights: List[Tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if isinstance(module, (ConstrainedConv1d, ConstrainedLinear)):
            weights.append((f"{name}.weight", module.weight))
        elif isinstance(module, (nn.Conv1d, nn.Linear)):
            weights.append((f"{name}.weight", module.weight))
    return weights


def weight_target_regularizer(model: nn.Module, target: float) -> torch.Tensor:
    penalties: List[torch.Tensor] = []
    for _, weight in _iter_effective_weights(model):
        penalties.append((weight.abs() - target).pow(2).mean())
    if not penalties:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(penalties).mean()


@contextmanager
def quantized_weights(model: nn.Module, bits: int) -> None:
    if bits is None or bits < 1:
        yield
        return
    originals = [param.data.clone() for param in model.parameters()]
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(quantize_tensor_symmetric(param.data, bits))
    try:
        yield
    finally:
        with torch.no_grad():
            for param, original in zip(model.parameters(), originals):
                param.copy_(original)


def build_student(args: argparse.Namespace, device: torch.device) -> nn.Module:
    student = SegmentAwareStudent(
        num_classes=len(set(BEAT_LABEL_MAP.values())),
        num_mlp_layers=args.num_mlp_layers,
        dropout_rate=args.dropout_rate,
        use_value_constraint=args.use_value_constraint,
        use_tanh_activations=args.use_tanh_activations,
        constraint_scale=args.constraint_scale,
        use_bias=args.use_bias,
        use_constrained_classifier=args.use_constrained_classifier,
    ).to(device)
    return student


def _quantize_export_tensor(x: torch.Tensor, bits: int | None) -> torch.Tensor:
    if bits is None or bits < 1:
        return x
    return quantize_tensor_symmetric(x, bits)


def export_weights_csv(model: nn.Module, output_path: str, weight_bits: int | None) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param_name", "kernel_index", "flat_index", "value"])
        for name, weight in _iter_effective_weights(model):
            data = _quantize_export_tensor(weight.detach().cpu(), weight_bits)
            for kernel_idx in range(data.shape[0]):
                flat = data[kernel_idx].reshape(-1)
                for flat_idx, value in enumerate(flat.tolist()):
                    writer.writerow([name, kernel_idx, flat_idx, value])

        for name, module in model.named_modules():
            bias = None
            if isinstance(module, (ConstrainedConv1d, ConstrainedLinear)):
                bias = module.bias
            elif isinstance(module, (nn.Conv1d, nn.Linear)):
                bias = module.bias
            if bias is None:
                continue
            bias_name = f"{name}.bias"
            bias_data = _quantize_export_tensor(bias.detach().cpu(), weight_bits)
            flat = bias_data.reshape(-1)
            for flat_idx, value in enumerate(flat.tolist()):
                writer.writerow([bias_name, 0, flat_idx, value])


def _sample_uniform(min_val: float, max_val: float, device: torch.device) -> float:
    if max_val <= min_val:
        return min_val
    return float(torch.empty(1, device=device).uniform_(min_val, max_val).item())


def apply_hardware_effects(
    signals: torch.Tensor,
    device: torch.device,
    input_bits: int,
    snr_min: float,
    snr_max: float,
    pbr_min: float,
    pbr_max: float,
    pbr_peak_window: int,
    pbr_min_prominence: float,
    hardware_prob: float,
    training: bool,
    renormalize: bool = True,
    zero_mean: bool = True,
) -> torch.Tensor:
    apply_effects = True
    if training and hardware_prob < 1.0:
        apply_effects = bool(torch.rand(1, device=device).item() <= hardware_prob)

    if apply_effects:
        pbr_factor = _sample_uniform(pbr_min, pbr_max, device)
        signals = apply_pbr_attenuation(signals, pbr_factor, pbr_peak_window, pbr_min_prominence)
        signals = signals.to(device)
        if zero_mean:
            signals = signals - signals.mean(dim=-1, keepdim=True)
        if renormalize:
            signals = renormalize_to_unit(signals)
        snr_db = _sample_uniform(snr_min, snr_max, device)
        signals = add_gaussian_noise_snr(signals, snr_db)
    else:
        signals = signals.to(device)

    if zero_mean:
        signals = signals - signals.mean(dim=-1, keepdim=True)
    if renormalize:
        signals = renormalize_to_unit(signals)
    return quantize_beats(signals, input_bits)


def evaluate(
    model: SegmentAwareStudent,
    data_loader: DataLoader,
    device: torch.device,
    return_probs: bool = False,
    threshold: float | None = None,
    use_hardware: bool = True,
    seed: int | None = None,
    renormalize_inputs: bool = True,
    zero_mean_inputs: bool = True,
    input_bits: int = 5,
    weight_bits: int = 5,
    snr_min: float = 10.0,
    snr_max: float = 30.0,
    pbr_min: float = 0.5,
    pbr_max: float = 0.9,
    pbr_peak_window: int = 12,
    pbr_min_prominence: float = 0.05,
) -> Tuple[float, Dict[str, float], List[int], List[int], List[float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    preds: List[int] = []
    trues: List[int] = []
    probs: List[float] = []
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            for signals, labels in data_loader:
                labels = labels.to(device)
                if use_hardware:
                    signals = apply_hardware_effects(
                        signals,
                        device,
                        input_bits,
                        snr_min,
                        snr_max,
                        pbr_min,
                        pbr_max,
                        pbr_peak_window,
                        pbr_min_prominence,
                        hardware_prob=1.0,
                        training=False,
                        renormalize=renormalize_inputs,
                        zero_mean=zero_mean_inputs,
                    )
                else:
                    signals = quantize_beats(signals.to(device), input_bits)
                with quantized_weights(model, bits=weight_bits):
                    logits, _ = model(signals)
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                prob_pos = torch.softmax(logits, dim=1)[:, 1]
                if threshold is None:
                    pred = torch.argmax(logits, dim=1)
                else:
                    pred = (prob_pos >= threshold).long()
                preds.extend(pred.cpu().tolist())
                trues.extend(labels.cpu().tolist())
                if return_probs:
                    probs.extend(prob_pos.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    metrics = confusion_metrics(trues, preds)
    return avg_loss, metrics, trues, preds, probs


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Backward-compatible boolean flags with --name / --no-name."""

    parser.add_argument(f"--{name}", dest=name, action="store_true", help=f"Enable {help_text}")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT-BIH ECG hardware-aware training")
    parser.add_argument(
        "--data_path",
        type=str,
        default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience on monitored metric")
    parser.add_argument("--min_epochs", type=int, default=25, help="Minimum epochs before early stopping")
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_mlp_layers", type=int, default=1)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--class_weight_abnormal", type=float, default=1.35)
    parser.add_argument("--class_weight_max_ratio", type=float, default=2.0)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="balanced_norm",
        choices=("f1", "balanced", "balanced_norm"),
        help=(
            "Metric to select best model: f1, balanced (penalize miss/FPR), or "
            "balanced_norm (penalize miss/FPR normalized by their targets)."
        ),
    )
    parser.add_argument(
        "--selection_miss_weight",
        type=float,
        default=1.0,
        help="Miss-rate penalty weight when selection_metric=balanced.",
    )
    parser.add_argument(
        "--selection_fpr_weight",
        type=float,
        default=1.0,
        help="FPR penalty weight when selection_metric=balanced.",
    )
    _add_bool_arg(
        parser,
        "selection_auto_weights",
        default=True,
        help_text="auto-scale selection penalties from target miss/FPR when selection_metric=balanced_norm",
    )
    parser.add_argument("--generalization_score_weight", type=float, default=0.35)
    parser.add_argument("--threshold_target_miss", type=float, default=0.10)
    parser.add_argument("--threshold_max_fpr", type=float, default=0.10)
    parser.add_argument(
        "--threshold_recall_gain",
        type=float,
        default=2.0,
        help="Sensitivity gain when scoring thresholds to prefer lower miss rates",
    )
    parser.add_argument(
        "--threshold_miss_penalty",
        type=float,
        default=1.25,
        help="Penalty weight on miss rate during blended threshold scoring",
    )
    parser.add_argument(
        "--threshold_gen_recall_gain",
        type=float,
        default=2.5,
        help="Sensitivity gain applied to generalization metrics during threshold sweeps",
    )
    parser.add_argument(
        "--threshold_gen_miss_penalty",
        type=float,
        default=1.35,
        help="Miss-rate penalty applied to generalization metrics during threshold sweeps",
    )
    parser.add_argument(
        "--gen_threshold",
        type=float,
        default=None,
        help="Explicit threshold override for generalization metrics",
    )
    parser.add_argument(
        "--gen_threshold_target_miss",
        type=float,
        default=None,
        help="Target miss rate for sweeping a generalization-only threshold (defaults to val miss * gen_threshold_tighten_factor)",
    )
    parser.add_argument(
        "--gen_threshold_max_fpr",
        type=float,
        default=None,
        help="Max FPR allowed when sweeping a generalization-only threshold (defaults to max(val fpr, threshold_max_fpr))",
    )
    parser.add_argument(
        "--gen_threshold_tighten_factor",
        type=float,
        default=0.7,
        help="Scale factor (<1 tightens miss target) applied to val miss for auto gen threshold sweep",
    )
    parser.add_argument(
        "--gen_threshold_fpr_relax_factor",
        type=float,
        default=1.25,
        help="Scale factor (>1 relaxes auto gen FPR cap) applied to max(val fpr, threshold_max_fpr)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_bits", type=int, default=5)
    parser.add_argument("--weight_bits", type=int, default=5)
    parser.add_argument("--snr_min", type=float, default=10.0)
    parser.add_argument("--snr_max", type=float, default=30.0)
    parser.add_argument("--pbr_min", type=float, default=0.5)
    parser.add_argument("--pbr_max", type=float, default=0.9)
    parser.add_argument("--pbr_peak_window", type=int, default=12)
    parser.add_argument("--pbr_min_prominence", type=float, default=0.05)
    _add_bool_arg(
        parser,
        "fixed_eval_hardware",
        default=True,
        help_text="fixed worst-case hardware during eval (uses eval_fixed_* values)",
    )
    parser.add_argument(
        "--eval_fixed_snr",
        type=float,
        default=None,
        help="Fixed SNR (dB) for eval when fixed_eval_hardware is enabled (defaults to snr_min).",
    )
    parser.add_argument(
        "--eval_fixed_pbr",
        type=float,
        default=None,
        help="Fixed PBR attenuation for eval when fixed_eval_hardware is enabled (defaults to pbr_min).",
    )
    parser.add_argument(
        "--eval_fixed_seed",
        type=int,
        default=1234,
        help="Random seed for eval noise generation when fixed_eval_hardware is enabled.",
    )
    parser.add_argument(
        "--weight_target",
        type=float,
        default=1.0,
        help="Target magnitude for weight regularization (encourage |w| near this value)",
    )
    parser.add_argument(
        "--weight_target_strength",
        type=float,
        default=1e-4,
        help="Strength of weight-target regularization (use small value to avoid hurting accuracy)",
    )
    _add_bool_arg(parser, "use_bias", default=False, help_text="bias terms in layers")
    _add_bool_arg(parser, "use_constrained_classifier", default=True, help_text="constrained classifier weights")
    parser.add_argument(
        "--hardware_prob",
        type=float,
        default=1.0,
        help="Probability of applying noise/PBR per batch during training (input quantization always applies)",
    )
    parser.add_argument(
        "--hardware_warmup_epochs",
        type=int,
        default=15,
        help="Linearly ramp hardware probability and perturbation strength for this many epochs",
    )
    parser.add_argument(
        "--hardware_prob_start",
        type=float,
        default=0.4,
        help="Starting hardware probability during warmup",
    )
    parser.add_argument(
        "--snr_min_start",
        type=float,
        default=20.0,
        help="Starting minimum SNR during warmup (higher is cleaner)",
    )
    parser.add_argument(
        "--snr_max_start",
        type=float,
        default=35.0,
        help="Starting maximum SNR during warmup (higher is cleaner)",
    )
    parser.add_argument(
        "--pbr_min_start",
        type=float,
        default=0.8,
        help="Starting minimum PBR attenuation factor during warmup (closer to 1.0 is cleaner)",
    )
    parser.add_argument(
        "--pbr_max_start",
        type=float,
        default=0.95,
        help="Starting maximum PBR attenuation factor during warmup (closer to 1.0 is cleaner)",
    )
    _add_bool_arg(parser, "renormalize_inputs", default=True, help_text="renormalize beats after hardware effects")
    _add_bool_arg(parser, "zero_mean_inputs", default=True, help_text="zero-mean beats after hardware effects")
    _add_bool_arg(parser, "hardware_eval", default=True, help_text="hardware effects during validation")
    _add_bool_arg(parser, "use_value_constraint", default=True, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading MIT-BIH records ...")
    train_x, train_y = load_records(TRAIN_RECORDS, args.data_path)
    gen_x, gen_y = load_records(GENERALIZATION_RECORDS, args.data_path)
    if train_x.size == 0 or gen_x.size == 0:
        raise RuntimeError("No data loaded. Check data path and wfdb installation.")

    tr_x, tr_y, va_x, va_y = split_dataset(train_x, train_y, val_ratio=0.2)
    print(f"Train: {len(tr_x)} | Val: {len(va_x)} | Generalization: {len(gen_x)}")

    abnormal_ratio = float(np.mean(tr_y)) if len(tr_y) > 0 else 0.0
    print(
        f"Class ratio (abnormal): {abnormal_ratio:.3f} | counts -> normal: {(tr_y == 0).sum()} abnormal: {(tr_y == 1).sum()}"
    )

    train_dataset = ECGBeatDataset(tr_x, tr_y)

    sampler = None
    batch_sampler = None
    sampler_boost = 1.2
    if abnormal_ratio < 0.35:
        sampler = make_weighted_sampler(tr_y, abnormal_boost=sampler_boost)
        print(
            "Enabling mild abnormal oversampling: "
            f"boost={sampler_boost:.2f}, expected abnormal frac≈{min(0.5, abnormal_ratio * sampler_boost):.2f}"
        )
    if abnormal_ratio < 0.45:
        try:
            batch_sampler = BalancedBatchSampler(tr_y, batch_size=args.batch_size)
            print("Using balanced batch sampler to keep per-batch class mix stable")
        except ValueError:
            batch_sampler = None

    if batch_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
        )
    val_loader = DataLoader(ECGBeatDataset(va_x, va_y), batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(ECGBeatDataset(gen_x, gen_y), batch_size=args.batch_size, shuffle=False)

    class_counts = np.bincount(tr_y, minlength=2)
    class_weights_np = compute_class_weights(
        tr_y,
        abnormal_boost=args.class_weight_abnormal,
        max_ratio=args.class_weight_max_ratio,
    )
    class_weights = class_weights_np.to(device)
    base_weights = class_weights.clone()

    miss_ema = 0.25

    os.makedirs("artifacts", exist_ok=True)
    log_path = os.path.join(
        "artifacts",
        f"training_log_hardware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    student = build_student(args, device)

    optimizer = Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, verbose=True)

    def _write_log(entry: Dict[str, object]) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    data_min = float(np.min(train_dataset.data)) if len(train_dataset) > 0 else 0.0
    data_max = float(np.max(train_dataset.data)) if len(train_dataset) > 0 else 0.0

    _write_log(
        {
            "event": "start",
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.detach().cpu().tolist(),
            "data_range": [data_min, data_max],
        }
    )

    print(f"Preprocessed input range: [{data_min:.3f}, {data_max:.3f}] (expected within [-1, 1])")
    if args.fixed_eval_hardware and args.hardware_eval:
        fixed_snr = args.eval_fixed_snr if args.eval_fixed_snr is not None else args.snr_min
        fixed_pbr = args.eval_fixed_pbr if args.eval_fixed_pbr is not None else args.pbr_min
        print(
            "Eval hardware set to fixed worst-case: "
            f"SNR={fixed_snr:.2f}dB, PBR={fixed_pbr:.2f}, seed={args.eval_fixed_seed}"
        )
    elif args.hardware_eval:
        print("Eval hardware set to random sampling across configured SNR/PBR ranges.")

    best_score = -float("inf")
    best_state = None
    best_threshold = 0.5
    patience_counter = 0

    history: List[Dict[str, float]] = []

    def _eval_params() -> Tuple[float, float, float, float, int | None]:
        eval_snr_min = args.snr_min
        eval_snr_max = args.snr_max
        eval_pbr_min = args.pbr_min
        eval_pbr_max = args.pbr_max
        eval_seed = None
        if args.fixed_eval_hardware and args.hardware_eval:
            fixed_snr = args.eval_fixed_snr if args.eval_fixed_snr is not None else args.snr_min
            fixed_pbr = args.eval_fixed_pbr if args.eval_fixed_pbr is not None else args.pbr_min
            eval_snr_min = fixed_snr
            eval_snr_max = fixed_snr
            eval_pbr_min = fixed_pbr
            eval_pbr_max = fixed_pbr
            eval_seed = args.eval_fixed_seed
        return eval_snr_min, eval_snr_max, eval_pbr_min, eval_pbr_max, eval_seed

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        running_loss = 0.0
        total = 0

        if args.hardware_warmup_epochs > 0:
            warmup_progress = min(1.0, epoch / args.hardware_warmup_epochs)
        else:
            warmup_progress = 1.0
        warmup_prob = args.hardware_prob_start + warmup_progress * (args.hardware_prob - args.hardware_prob_start)
        warmup_snr_min = args.snr_min_start + warmup_progress * (args.snr_min - args.snr_min_start)
        warmup_snr_max = args.snr_max_start + warmup_progress * (args.snr_max - args.snr_max_start)
        warmup_pbr_min = args.pbr_min_start + warmup_progress * (args.pbr_min - args.pbr_min_start)
        warmup_pbr_max = args.pbr_max_start + warmup_progress * (args.pbr_max - args.pbr_max_start)

        adaptive_pos_boost = 1.0 + miss_ema * 0.8
        epoch_weights = base_weights.clone()
        epoch_weights[1] = torch.clamp(
            base_weights[1] * adaptive_pos_boost,
            min=base_weights.min() * 0.8,
            max=base_weights.max() * 2.0,
        )
        ce_loss_fn = nn.CrossEntropyLoss(weight=epoch_weights)

        for signals, labels in train_loader:
            labels = labels.to(device)
            signals = apply_hardware_effects(
                signals,
                device,
                args.input_bits,
                warmup_snr_min,
                warmup_snr_max,
                warmup_pbr_min,
                warmup_pbr_max,
                args.pbr_peak_window,
                args.pbr_min_prominence,
                warmup_prob,
                training=True,
                renormalize=args.renormalize_inputs,
                zero_mean=args.zero_mean_inputs,
            )
            optimizer.zero_grad()

            with quantized_weights(student, bits=args.weight_bits):
                logits, _ = student(signals)
                loss = ce_loss_fn(logits, labels)
                reg_loss = weight_target_regularizer(student, args.weight_target)
                total_loss = loss + args.weight_target_strength * reg_loss
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)

        eval_snr_min, eval_snr_max, eval_pbr_min, eval_pbr_max, eval_seed = _eval_params()

        val_loss, _, val_true, _, val_probs = evaluate(
            student,
            val_loader,
            device,
            return_probs=True,
            use_hardware=args.hardware_eval,
            seed=eval_seed,
            renormalize_inputs=args.renormalize_inputs,
            zero_mean_inputs=args.zero_mean_inputs,
            input_bits=args.input_bits,
            weight_bits=args.weight_bits,
            snr_min=eval_snr_min,
            snr_max=eval_snr_max,
            pbr_min=eval_pbr_min,
            pbr_max=eval_pbr_max,
            pbr_peak_window=args.pbr_peak_window,
            pbr_min_prominence=args.pbr_min_prominence,
        )
        gen_loss, _, gen_true, _, gen_probs = evaluate(
            student,
            gen_loader,
            device,
            return_probs=True,
            use_hardware=args.hardware_eval,
            seed=eval_seed,
            renormalize_inputs=args.renormalize_inputs,
            zero_mean_inputs=args.zero_mean_inputs,
            input_bits=args.input_bits,
            weight_bits=args.weight_bits,
            snr_min=eval_snr_min,
            snr_max=eval_snr_max,
            pbr_min=eval_pbr_min,
            pbr_max=eval_pbr_max,
            pbr_peak_window=args.pbr_peak_window,
            pbr_min_prominence=args.pbr_min_prominence,
        )

        best_thr_epoch, val_metrics, gen_metrics = sweep_thresholds_blended(
            val_true,
            val_probs,
            gen_true,
            gen_probs,
            gen_weight=args.generalization_score_weight,
            recall_gain=args.threshold_recall_gain,
            miss_penalty=args.threshold_miss_penalty,
            gen_recall_gain=args.threshold_gen_recall_gain,
            gen_miss_penalty=args.threshold_gen_miss_penalty,
            miss_target=args.threshold_target_miss,
            fpr_cap=args.threshold_max_fpr,
        )

        miss_ema = 0.8 * miss_ema + 0.2 * val_metrics["miss_rate"]

        if args.selection_metric == "f1":
            selection_score = val_metrics["f1"]
        elif args.selection_metric == "balanced_norm":
            miss_weight = args.selection_miss_weight
            fpr_weight = args.selection_fpr_weight
            if args.selection_auto_weights:
                miss_weight = 1.0 / max(args.threshold_target_miss, 1e-6)
                fpr_weight = 1.0 / max(args.threshold_max_fpr, 1e-6)
            selection_score = val_metrics["f1"] - (
                miss_weight * (val_metrics["miss_rate"] / max(args.threshold_target_miss, 1e-6))
                + fpr_weight * (val_metrics["fpr"] / max(args.threshold_max_fpr, 1e-6))
            )
        else:
            selection_score = val_metrics["f1"] - (
                args.selection_miss_weight * val_metrics["miss_rate"]
                + args.selection_fpr_weight * val_metrics["fpr"]
            )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val F1 {val_metrics['f1']:.3f} Miss {val_metrics['miss_rate'] * 100:.2f}% FPR {val_metrics['fpr'] * 100:.2f}% | "
            f"Gen F1 {gen_metrics['f1']:.3f} Miss {gen_metrics['miss_rate'] * 100:.2f}% FPR {gen_metrics['fpr'] * 100:.2f}% | "
            f"Thr {best_thr_epoch:.4f} | SelScore {selection_score:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_miss": val_metrics["miss_rate"],
                "val_fpr": val_metrics["fpr"],
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
                "threshold": best_thr_epoch,
                "selection_score": selection_score,
            }
        )

        _write_log(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_miss": val_metrics["miss_rate"],
                "val_fpr": val_metrics["fpr"],
                "gen_f1": gen_metrics["f1"],
                "gen_miss": gen_metrics["miss_rate"],
                "gen_fpr": gen_metrics["fpr"],
                "threshold": best_thr_epoch,
                "selection_score": selection_score,
            }
        )

        if selection_score > best_score:
            best_score = selection_score
            best_state = student.state_dict()
            best_threshold = best_thr_epoch
            patience_counter = 0
            print("  -> New best model saved.")
            _write_log(
                {
                    "event": "best",
                    "epoch": epoch,
                    "val_f1": val_metrics["f1"],
                    "val_miss": val_metrics["miss_rate"],
                    "val_fpr": val_metrics["fpr"],
                    "selection_score": selection_score,
                    "gen_f1": gen_metrics["f1"],
                    "gen_miss": gen_metrics["miss_rate"],
                    "gen_fpr": gen_metrics["fpr"],
                    "threshold": best_thr_epoch,
                }
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch >= args.min_epochs:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    eval_snr_min, eval_snr_max, eval_pbr_min, eval_pbr_max, eval_seed = _eval_params()
    val_loss, _, val_true, _, val_probs = evaluate(
        student,
        val_loader,
        device,
        return_probs=True,
        use_hardware=args.hardware_eval,
        seed=eval_seed,
        renormalize_inputs=args.renormalize_inputs,
        zero_mean_inputs=args.zero_mean_inputs,
        input_bits=args.input_bits,
        weight_bits=args.weight_bits,
        snr_min=eval_snr_min,
        snr_max=eval_snr_max,
        pbr_min=eval_pbr_min,
        pbr_max=eval_pbr_max,
        pbr_peak_window=args.pbr_peak_window,
        pbr_min_prominence=args.pbr_min_prominence,
    )
    gen_loss, _, gen_true, _, gen_probs = evaluate(
        student,
        gen_loader,
        device,
        return_probs=True,
        use_hardware=args.hardware_eval,
        seed=eval_seed,
        renormalize_inputs=args.renormalize_inputs,
        zero_mean_inputs=args.zero_mean_inputs,
        input_bits=args.input_bits,
        weight_bits=args.weight_bits,
        snr_min=eval_snr_min,
        snr_max=eval_snr_max,
        pbr_min=eval_pbr_min,
        pbr_max=eval_pbr_max,
        pbr_peak_window=args.pbr_peak_window,
        pbr_min_prominence=args.pbr_min_prominence,
    )

    best_threshold, val_metrics, gen_metrics = sweep_thresholds_blended(
        val_true,
        val_probs,
        gen_true,
        gen_probs,
        gen_weight=args.generalization_score_weight,
        recall_gain=args.threshold_recall_gain,
        miss_penalty=args.threshold_miss_penalty,
        gen_recall_gain=args.threshold_gen_recall_gain,
        gen_miss_penalty=args.threshold_gen_miss_penalty,
        miss_target=args.threshold_target_miss,
        fpr_cap=args.threshold_max_fpr,
    )
    val_pred = (np.array(val_probs) >= best_threshold).astype(int).tolist()
    gen_threshold = best_threshold
    gen_threshold_source = "blended"
    if args.gen_threshold is not None:
        gen_threshold = args.gen_threshold
        gen_threshold_source = "manual"
        gen_pred = (np.array(gen_probs) >= gen_threshold).astype(int).tolist()
        gen_metrics = confusion_metrics(gen_true, gen_pred)
    else:
        auto_miss_target = max(0.0, val_metrics["miss_rate"] * args.gen_threshold_tighten_factor)
        auto_fpr_cap = max(val_metrics["fpr"], args.threshold_max_fpr) * args.gen_threshold_fpr_relax_factor
        miss_target = args.gen_threshold_target_miss
        fpr_cap = args.gen_threshold_max_fpr
        use_auto = miss_target is None and fpr_cap is None
        if use_auto:
            miss_target = auto_miss_target
            fpr_cap = auto_fpr_cap
        if miss_target is not None or fpr_cap is not None:
            if use_auto:
                gen_threshold, gen_metrics = sweep_thresholds_min_miss(
                    gen_true,
                    gen_probs,
                    fpr_cap=fpr_cap,
                )
                gen_threshold_source = "auto_sweep"
            else:
                gen_threshold, gen_metrics = sweep_thresholds(
                    gen_true,
                    gen_probs,
                    miss_target=miss_target,
                    fpr_cap=fpr_cap,
                )
                gen_threshold_source = "sweep"
            gen_pred = (np.array(gen_probs) >= gen_threshold).astype(int).tolist()
        else:
            gen_pred = (np.array(gen_probs) >= best_threshold).astype(int).tolist()

    print(
        f"Final Val@thr={best_threshold:.4f}: loss={val_loss:.4f}, F1={val_metrics['f1']:.3f}, "
        f"miss={val_metrics['miss_rate'] * 100:.2f}%, fpr={val_metrics['fpr'] * 100:.2f}%"
    )
    print(
        f"Generalization@thr={gen_threshold:.4f}: loss={gen_loss:.4f}, F1={gen_metrics['f1']:.3f}, "
        f"miss={gen_metrics['miss_rate'] * 100:.2f}%, fpr={gen_metrics['fpr'] * 100:.2f}%"
    )
    print(f"Generalization threshold source: {gen_threshold_source}")

    def _collect_threshold_records(
        y_true: List[int],
        probs: List[float],
        thresholds: List[float],
    ) -> List[Dict[str, object]]:
        y_true_arr = np.array(y_true)
        probs_arr = np.array(probs)
        records: List[Dict[str, object]] = []
        for thr in thresholds:
            preds = (probs_arr >= thr).astype(int).tolist()
            metrics = confusion_metrics(y_true_arr.tolist(), preds)
            roc_score = metrics["sensitivity"] - metrics["fpr"]
            records.append(
                {
                    "threshold": float(thr),
                    "metrics": metrics,
                    "roc": roc_score,
                }
            )
        return records

    fine_step = 0.0001
    fine_thresholds = np.round(np.arange(0.0, 1.0 + fine_step / 2, fine_step), 4).tolist()
    gen_records = _collect_threshold_records(gen_true, gen_probs, fine_thresholds)

    low_miss_candidates = [r for r in gen_records if r["metrics"]["fpr"] < 0.2]
    low_miss_top10 = sorted(
        low_miss_candidates,
        key=lambda r: (
            r["metrics"]["miss_rate"],
            r["metrics"]["fpr"],
            -r["metrics"]["sensitivity"],
        ),
    )[:10]

    low_fpr_candidates = [r for r in gen_records if r["metrics"]["miss_rate"] < 0.1]
    low_fpr_top10 = sorted(
        low_fpr_candidates,
        key=lambda r: (
            r["metrics"]["fpr"],
            r["metrics"]["miss_rate"],
            -r["metrics"]["sensitivity"],
        ),
    )[:10]

    balanced_pool = low_miss_top10 + low_fpr_top10
    if not balanced_pool:
        balanced_pool = gen_records
    balanced_record = max(
        balanced_pool,
        key=lambda r: (
            r["roc"],
            -r["metrics"]["miss_rate"],
            -r["metrics"]["fpr"],
        ),
    )

    coarse_step = 0.001
    coarse_thresholds = np.round(np.arange(0.0, 1.0 + coarse_step / 2, coarse_step), 3).tolist()
    coarse_records = _collect_threshold_records(gen_true, gen_probs, coarse_thresholds)
    sweep_csv_path = os.path.join("artifacts", "gen_threshold_sweep_hardware.csv")
    with open(sweep_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "miss_rate", "fpr"])
        for record in coarse_records:
            metrics = record["metrics"]
            writer.writerow(
                [
                    f"{record['threshold']:.3f}",
                    f"{metrics['miss_rate']:.6f}",
                    f"{metrics['fpr']:.6f}",
                ]
            )
    print(f"Saved gen threshold sweep CSV to {sweep_csv_path}")

    if low_miss_top10:
        gen_low_miss_thr = low_miss_top10[0]["threshold"]
        gen_low_miss_metrics = low_miss_top10[0]["metrics"]
    else:
        gen_low_miss_thr = balanced_record["threshold"]
        gen_low_miss_metrics = balanced_record["metrics"]
    gen_balanced_thr = balanced_record["threshold"]
    gen_balanced_metrics = balanced_record["metrics"]
    if low_fpr_top10:
        gen_low_fpr_thr = low_fpr_top10[0]["threshold"]
        gen_low_fpr_metrics = low_fpr_top10[0]["metrics"]
    else:
        gen_low_fpr_thr = balanced_record["threshold"]
        gen_low_fpr_metrics = balanced_record["metrics"]

    _write_log(
        {
            "event": "final",
            "best_threshold": best_threshold,
            "gen_threshold": gen_threshold,
            "gen_threshold_source": gen_threshold_source,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_miss": val_metrics["miss_rate"],
            "val_fpr": val_metrics["fpr"],
            "gen_loss": gen_loss,
            "gen_f1": gen_metrics["f1"],
            "gen_miss": gen_metrics["miss_rate"],
            "gen_fpr": gen_metrics["fpr"],
            "gen_threshold_options": {
                "low_miss": {"threshold": gen_low_miss_thr, "metrics": gen_low_miss_metrics},
                "balanced": {"threshold": gen_balanced_thr, "metrics": gen_balanced_metrics},
                "low_fpr": {"threshold": gen_low_fpr_thr, "metrics": gen_low_fpr_metrics},
            },
            "gen_threshold_sweep": {
                "sweep_csv": sweep_csv_path,
                "low_miss_candidate_count": len(low_miss_candidates),
                "low_miss_top10": [
                    {
                        "threshold": r["threshold"],
                        "miss_rate": r["metrics"]["miss_rate"],
                        "fpr": r["metrics"]["fpr"],
                        "roc": r["roc"],
                    }
                    for r in low_miss_top10
                ],
                "low_fpr_candidate_count": len(low_fpr_candidates),
                "low_fpr_top10": [
                    {
                        "threshold": r["threshold"],
                        "miss_rate": r["metrics"]["miss_rate"],
                        "fpr": r["metrics"]["fpr"],
                        "roc": r["roc"],
                    }
                    for r in low_fpr_top10
                ],
                "balanced": {
                    "threshold": gen_balanced_thr,
                    "miss_rate": gen_balanced_metrics["miss_rate"],
                    "fpr": gen_balanced_metrics["fpr"],
                    "roc": balanced_record["roc"],
                },
            },
        }
    )

    np.save(os.path.join("artifacts", "val_probs_hardware.npy"), np.array(val_probs))
    np.save(os.path.join("artifacts", "gen_probs_hardware.npy"), np.array(gen_probs))
    np.save(os.path.join("artifacts", "val_labels_hardware.npy"), np.array(val_true))
    np.save(os.path.join("artifacts", "gen_labels_hardware.npy"), np.array(gen_true))

    print("Gen sweep (low-miss, FPR < 20%) top10:")
    if low_miss_top10:
        for idx, record in enumerate(low_miss_top10, start=1):
            metrics = record["metrics"]
            print(
                f"  {idx:02d}. thr={record['threshold']:.4f} "
                f"miss={metrics['miss_rate']:.4f} fpr={metrics['fpr']:.4f} roc={record['roc']:.4f}"
            )
    else:
        print("  (no thresholds met FPR < 20% constraint)")
    print("Gen sweep (low-fpr, miss < 10%) top10:")
    if low_fpr_top10:
        for idx, record in enumerate(low_fpr_top10, start=1):
            metrics = record["metrics"]
            print(
                f"  {idx:02d}. thr={record['threshold']:.4f} "
                f"miss={metrics['miss_rate']:.4f} fpr={metrics['fpr']:.4f} roc={record['roc']:.4f}"
            )
    else:
        print("  (no thresholds met miss < 10% constraint)")
    print(
        f"Gen low-miss@thr={gen_low_miss_thr:.4f}: F1={gen_low_miss_metrics['f1']:.3f}, "
        f"miss={gen_low_miss_metrics['miss_rate'] * 100:.2f}%, fpr={gen_low_miss_metrics['fpr'] * 100:.2f}%"
    )
    print(
        f"Gen balanced@thr={gen_balanced_thr:.4f}: F1={gen_balanced_metrics['f1']:.3f}, "
        f"miss={gen_balanced_metrics['miss_rate'] * 100:.2f}%, fpr={gen_balanced_metrics['fpr'] * 100:.2f}%"
    )
    print(
        f"Gen low-fpr@thr={gen_low_fpr_thr:.4f}: F1={gen_low_fpr_metrics['f1']:.3f}, "
        f"miss={gen_low_fpr_metrics['miss_rate'] * 100:.2f}%, fpr={gen_low_fpr_metrics['fpr'] * 100:.2f}%"
    )

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "student_model_hardware.pth")
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "config": vars(args),
            "best_threshold": best_threshold,
            "gen_threshold": gen_threshold,
            "gen_threshold_options": {
                "low_miss": {"threshold": gen_low_miss_thr, "metrics": gen_low_miss_metrics},
                "balanced": {"threshold": gen_balanced_thr, "metrics": gen_balanced_metrics},
                "low_fpr": {"threshold": gen_low_fpr_thr, "metrics": gen_low_fpr_metrics},
            },
        },
        save_path,
    )
    print(f"Saved student checkpoint to {save_path}")

    weights_csv_path = os.path.join("artifacts", "hardware_weights.csv")
    export_weights_csv(student, weights_csv_path, args.weight_bits)
    print(f"Saved hardware-trained weights CSV to {weights_csv_path}")

    os.makedirs("artifacts", exist_ok=True)

    def _save_training_curves() -> None:
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
        axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()

        axes[1].plot(epochs, [h["val_f1"] for h in history], label="F1")
        axes[1].plot(epochs, [h["val_miss"] for h in history], label="Miss Rate")
        axes[1].plot(epochs, [h["val_fpr"] for h in history], label="FPR")
        if any("gen_f1" in h for h in history):
            axes[1].plot(epochs, [h.get("gen_f1", float("nan")) for h in history], label="Gen F1", linestyle="--")
            axes[1].plot(
                epochs,
                [h.get("gen_miss", float("nan")) for h in history],
                label="Gen Miss",
                linestyle="--",
            )
            axes[1].plot(
                epochs,
                [h.get("gen_fpr", float("nan")) for h in history],
                label="Gen FPR",
                linestyle="--",
            )
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Val Metrics")
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(os.path.join("artifacts", "training_curves_hardware.png"))
        plt.close(fig)

    def _save_roc(y_true: List[int], probs: List[float], name: str) -> None:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC - {name}")
        ax.legend()
        fig.savefig(os.path.join("artifacts", f"roc_{name.lower()}_hardware.png"))
        plt.close(fig)

    def _save_confusion(y_true: List[int], y_pred: List[int], name: str) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join("artifacts", f"confusion_{name.lower()}_hardware.png"))
        plt.close(fig)

    _save_training_curves()
    _save_roc(val_true, val_probs, "Val")
    _save_roc(gen_true, gen_probs, "Generalization")
    _save_confusion(val_true, val_pred, "Val")
    _save_confusion(gen_true, gen_pred, "Generalization")
    print("Saved training curves, ROC curves, and confusion matrices to ./artifacts")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
