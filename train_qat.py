"""QAT fine-tuning with 5-bit activation/weight fake quantization and noise/PBR augmentation.

Adds curriculum warmup for quantization与增广、动态阈值重标定，以及带 warmup 的学习率调度，
便于在低 SNR/低 PBR + 量化场景下更平稳收敛。

本版进一步放缓 curriculum（先 FP32、弱增广，再逐步切换到 5-bit 与强噪声/PBR），
并改进阈值校准避免极端阈值导致的全正/全负预测。
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from utils import confusion_metrics
from train import GENERALIZATION_RECORDS, TRAIN_RECORDS


def fake_quant_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits is None or bits < 1:
        return x
    qmax = (2 ** (bits - 1)) - 1
    qmin = -2 ** (bits - 1)
    max_abs = x.detach().abs().max()
    scale = (max_abs / qmax).clamp(min=1e-8)
    return torch.fake_quantize_per_tensor_affine(x, float(scale), 0, int(qmin), int(qmax))


def clamp_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


def add_random_noise(signals: torch.Tensor, snr_min: float, snr_max: float) -> torch.Tensor:
    """Add per-sample Gaussian noise at a random SNR in [snr_min, snr_max] dB."""
    batch_size = signals.size(0)
    snr_db = torch.empty(batch_size, device=signals.device).uniform_(snr_min, snr_max)
    snr_linear = 10 ** (snr_db / 10.0)
    power = signals.pow(2).mean(dim=2, keepdim=True)
    noise_power = power / snr_linear.view(-1, 1, 1)
    noise_std = torch.sqrt(noise_power + 1e-12)
    noise = torch.randn_like(signals) * noise_std
    return signals + noise


def apply_random_pbr(signals: torch.Tensor, pbr_min: float, pbr_max: float) -> torch.Tensor:
    """Scale beat amplitude around its median to simulate PBR attenuation."""
    factors = torch.empty(signals.size(0), device=signals.device).uniform_(pbr_min, pbr_max).view(-1, 1, 1)
    baseline = signals.median(dim=2, keepdim=True).values
    return baseline + factors * (signals - baseline)


def interpolate_curriculum(start: float, end: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return end
    progress = min(max(epoch, 0), warmup_epochs) / float(warmup_epochs)
    return start + (end - start) * progress


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    epoch: int,
    max_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.1,
) -> float:
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
        lr = base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine.item())
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class QATStudent(SegmentAwareStudent):
    """Segment-aware student with per-forward fake quantization."""

    def __init__(self, *args, activation_bits: int = 5, weight_bits: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits

    def _quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        return fake_quant_symmetric(w, self.weight_bits)

    def _quant_act(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quant_symmetric(x, self.activation_bits)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = clamp_unit(x)
        x = self._quant_act(x)

        p_seg = x[:, :, 0:120]
        qrs_seg = x[:, :, 120:240]
        t_seg = x[:, :, 240:360]
        global_seg = x

        tokens: List[torch.Tensor] = []

        def _conv_token(seg: torch.Tensor, conv: nn.Module, num_blocks: int) -> torch.Tensor:
            w = self._quant_weight(conv.weight)
            b = conv.bias
            out = self._activate(F.conv1d(seg, w, bias=b, stride=conv.stride, padding=conv.padding))
            return self._pool_tokens(out, num_blocks)

        tokens.append(_conv_token(p_seg, self.conv_p, num_blocks=2))
        tokens.append(_conv_token(qrs_seg, self.conv_qrs, num_blocks=3))
        tokens.append(_conv_token(t_seg, self.conv_t, num_blocks=2))

        w_g = self._quant_weight(self.conv_global.weight)
        b_g = self.conv_global.bias
        g_out = self._activate(F.conv1d(global_seg, w_g, bias=b_g, stride=self.conv_global.stride, padding=self.conv_global.padding))
        g_token = g_out.mean(dim=-1, keepdim=True).transpose(1, 2)
        tokens.append(g_token)

        h0 = torch.cat(tokens, dim=1)

        h = h0
        for layer in self.mlp_layers:
            h = self._scale_if_needed(h)
            w_l = self._quant_weight(layer.weight)
            b_l = layer.bias
            h = self._activate(F.linear(h, w_l, b_l))
            h = self.token_dropout(h)
            h = self._quant_act(h)

        h_pool = h.mean(dim=1)
        h_pool = self.dropout(h_pool)
        w_cls = self._quant_weight(self.classifier.weight)
        b_cls = self.classifier.bias
        logits = F.linear(h_pool, w_cls, b_cls)
        return logits, h_pool


def build_student_from_checkpoint(init_path: str, device: torch.device, activation_bits: int, weight_bits: int) -> QATStudent:
    ckpt = torch.load(init_path, map_location=device)
    config = ckpt.get("config", {})
    student = QATStudent(
        num_classes=2,
        num_mlp_layers=int(config.get("num_mlp_layers", 1)),
        dropout_rate=float(config.get("dropout_rate", 0.0)),
        use_value_constraint=bool(config.get("use_value_constraint", True)),
        use_tanh_activations=bool(config.get("use_tanh_activations", False)),
        constraint_scale=float(config.get("constraint_scale", 1.0)),
        activation_bits=activation_bits,
        weight_bits=weight_bits,
    ).to(device)
    student.load_state_dict(ckpt["student_state_dict"], strict=False)
    return student


def evaluate(model: QATStudent, loader: DataLoader, device: torch.device, threshold: float) -> Tuple[float, Dict[str, float]]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    preds: List[int] = []
    trues: List[int] = []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits, _ = model(signals)
            loss = ce(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob_pos >= threshold).long()
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    metrics = confusion_metrics(trues, preds)
    return avg_loss, metrics


def train_one_epoch(
    model: QATStudent,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    snr_min: float,
    snr_max: float,
    pbr_min: float,
    pbr_max: float,
    activation_bits: int,
    weight_bits: int,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    running = 0.0
    total = 0
    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        signals = apply_random_pbr(signals, pbr_min, pbr_max)
        signals = add_random_noise(signals, snr_min, snr_max)
        signals = clamp_unit(signals)
        signals = torch.nan_to_num(signals)

        # 动态量化位宽（curriculum warmup）
        orig_act_bits = model.activation_bits
        orig_w_bits = model.weight_bits
        model.activation_bits = activation_bits
        model.weight_bits = weight_bits

        logits, _ = model(signals)
        logits = torch.nan_to_num(logits)
        loss = ce(logits, labels)
        if torch.isnan(loss):
            # 跳过异常 batch，避免梯度爆炸/NaN 传播
            model.activation_bits = orig_act_bits
            model.weight_bits = orig_w_bits
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.activation_bits = orig_act_bits
        model.weight_bits = orig_w_bits

        running += loss.item() * labels.size(0)
        total += labels.size(0)
    return running / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-bit QAT fine-tune with noise & PBR augmentation")
    parser.add_argument(
        "--data_path",
        type=str,
        default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/",
        help="Path to MIT-BIH dataset",
    )
    parser.add_argument("--init_checkpoint", type=str, default="saved_models/student_model.pth")
    parser.add_argument("--activation_bits", type=int, default=5)
    parser.add_argument("--weight_bits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.2, help="Cosine decay floor as fraction of base lr")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for LR and quant/aug curriculum")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.346)
    parser.add_argument("--snr_min", type=float, default=10.0)
    parser.add_argument("--snr_max", type=float, default=30.0)
    parser.add_argument("--pbr_min", type=float, default=0.5)
    parser.add_argument("--pbr_max", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_path", type=str, default="saved_models/student_model_qat5bit.pth")
    return parser.parse_args()


def calibrate_threshold(model: QATStudent, loader: DataLoader, device: torch.device, default_thr: float = 0.346) -> float:
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits, _ = model(signals)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(prob_pos.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 选择使 FNR+FPR 最小的阈值（致密 grid + 下限），避免全正/全负
    probs_tensor = torch.tensor(all_probs)
    labels_tensor = torch.tensor(all_labels)
    candidate = torch.linspace(0.05, 0.95, steps=91)
    best_thr = default_thr
    best_err = float("inf")
    for thr in candidate:
        pred = (probs_tensor >= thr).long()
        tn = ((pred == 0) & (labels_tensor == 0)).sum().item()
        fp = ((pred == 1) & (labels_tensor == 0)).sum().item()
        fn = ((pred == 0) & (labels_tensor == 1)).sum().item()
        tp = ((pred == 1) & (labels_tensor == 1)).sum().item()
        fnr = fn / max(fn + tp, 1)
        fpr = fp / max(fp + tn, 1)
        err = fnr + fpr
        if err < best_err:
            best_err = err
            best_thr = float(thr.item())
    # 避免阈值过低/过高导致极端预测
    return float(max(0.15, min(0.85, best_thr)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_labels = load_records(TRAIN_RECORDS, args.data_path)
    val_x, val_y = load_records(GENERALIZATION_RECORDS, args.data_path)
    if not train_data.size or not val_x.size:
        raise RuntimeError("Dataset loading failed; check data_path and record availability.")

    tr_x, tr_y, val_split_x, val_split_y = split_dataset(train_data, train_labels, val_ratio=0.1)

    train_ds: Dataset[Tuple[torch.Tensor, int]] = ECGBeatDataset(tr_x, tr_y)
    val_ds: Dataset[Tuple[torch.Tensor, int]] = ECGBeatDataset(val_split_x, val_split_y)
    gen_ds: Dataset[Tuple[torch.Tensor, int]] = ECGBeatDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False)

    model = build_student_from_checkpoint(args.init_checkpoint, device, args.activation_bits, args.weight_bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    current_threshold = args.threshold
    for epoch in range(1, args.max_epochs + 1):
        # curriculum：先 FP32/弱增广，再逐步放开量化和噪声/PBR
        act_bits = int(round(interpolate_curriculum(-1.0, args.activation_bits, epoch, args.warmup_epochs)))
        w_bits = int(round(interpolate_curriculum(-1.0, args.weight_bits, epoch, args.warmup_epochs)))
        snr_min = interpolate_curriculum(35.0, args.snr_min, epoch, args.warmup_epochs)
        snr_max = interpolate_curriculum(35.0, args.snr_max, epoch, args.warmup_epochs)
        pbr_min = interpolate_curriculum(0.95, args.pbr_min, epoch, args.warmup_epochs)
        pbr_max = interpolate_curriculum(0.95, args.pbr_max, epoch, args.warmup_epochs)
        lr_now = adjust_learning_rate(optimizer, args.lr, epoch, args.max_epochs, args.warmup_epochs, args.min_lr_ratio)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            snr_min,
            snr_max,
            pbr_min,
            pbr_max,
            act_bits,
            w_bits,
        )

        current_threshold = calibrate_threshold(model, val_loader, device, default_thr=args.threshold)
        val_loss, val_metrics = evaluate(model, val_loader, device, current_threshold)
        gen_loss, gen_metrics = evaluate(model, gen_loader, device, current_threshold)

        print(
            f"Epoch {epoch:02d} | lr={lr_now:.5f} act_bits={act_bits} w_bits={w_bits} "
            f"snr=[{snr_min:.1f},{snr_max:.1f}] pbr=[{pbr_min:.2f},{pbr_max:.2f}] "
            f"thr={current_threshold:.3f} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_fnr={val_metrics['miss_rate']:.4f} val_fpr={val_metrics['fpr']:.4f} "
            f"gen_fnr={gen_metrics['miss_rate']:.4f} gen_fpr={gen_metrics['fpr']:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "student_state_dict": model.state_dict(),
                "config": {
                    "num_mlp_layers": model.mlp_layers.__len__(),
                    "dropout_rate": float(model.dropout.p if isinstance(model.dropout, nn.Dropout) else 0.0),
                    "use_value_constraint": model.use_value_constraint,
                    "use_tanh_activations": model.use_tanh_activations,
                    "constraint_scale": 1.0,
                },
                "activation_bits": args.activation_bits,
                "weight_bits": args.weight_bits,
                "threshold": current_threshold,
            }

    if best_state is not None:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(best_state, args.save_path)
        print(f"Saved QAT checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
