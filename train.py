"""Training script for segment-aware ECG classification with knowledge distillation."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from constraints import ConstrainedLinear, scale_to_unit
from data import BEAT_LABEL_MAP, ECGBeatDataset, load_records, set_seed, split_dataset
from models.student import SegmentAwareStudent
from models.teacher import ResNet18_1D
from utils import compute_class_weights, confusion_metrics, kd_logit_loss, l2_normalize


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


def build_teacher(num_classes: int, embedding_dim: int, device: torch.device) -> nn.Module:
    teacher = ResNet18_1D(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def train_teacher(
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
) -> None:
    teacher.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(teacher.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=False)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, max_epochs + 1):
        teacher.train()
        running_loss = 0.0
        total = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = teacher(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        train_loss = running_loss / max(total, 1)

        # validation
        teacher.eval()
        val_loss = 0.0
        vtotal = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                logits, _ = teacher(signals)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                vtotal += labels.size(0)
        val_loss = val_loss / max(vtotal, 1)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = teacher.state_dict()
        if epoch % 1 == 0:
            print(f"[Teacher] Epoch {epoch:02d}/{max_epochs} TrainLoss {train_loss:.4f} ValLoss {val_loss:.4f}")

    if best_state is not None:
        teacher.load_state_dict(best_state)
    teacher.eval()


def build_student(args: argparse.Namespace, device: torch.device) -> nn.Module:
    student = SegmentAwareStudent(
        num_classes=len(set(BEAT_LABEL_MAP.values())),
        num_mlp_layers=args.num_mlp_layers,
        dropout_rate=args.dropout_rate,
        use_value_constraint=args.use_value_constraint,
        use_tanh_activations=args.use_tanh_activations,
        constraint_scale=args.constraint_scale,
    ).to(device)
    return student


def evaluate(model: SegmentAwareStudent, data_loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    preds = []
    trues = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    metrics = confusion_metrics(trues, preds)
    return avg_loss, metrics


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Backward-compatible boolean flags with --name / --no-name."""

    parser.add_argument(f"--{name}", dest=name, action="store_true", help=f"Enable {help_text}")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT-BIH ECG training with KD and constraints")
    parser.add_argument("--data_path", type=str, default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss")
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--num_mlp_layers", type=int, default=2)
    parser.add_argument("--class_weight_abnormal", type=float, default=1.3)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_embedding_dim", type=int, default=128)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_d", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--teacher_auto_train_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    _add_bool_arg(parser, "use_kd", default=True, help_text="knowledge distillation")
    _add_bool_arg(parser, "use_value_constraint", default=False, help_text="value-constrained weights/activations")
    _add_bool_arg(parser, "use_tanh_activations", default=False, help_text="tanh activations before constrained layers")
    _add_bool_arg(parser, "auto_train_teacher", default=True, help_text="auto teacher training when no checkpoint is provided")

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

    train_loader = DataLoader(ECGBeatDataset(tr_x, tr_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ECGBeatDataset(va_x, va_y), batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(ECGBeatDataset(gen_x, gen_y), batch_size=args.batch_size, shuffle=False)

    student = build_student(args, device)

    class_weights = compute_class_weights(tr_y, abnormal_boost=args.class_weight_abnormal).to(device)

    teacher = None
    proj_T: nn.Module
    proj_S: nn.Module
    kd_enabled = args.use_kd
    auto_teacher_path = os.path.join("saved_models", "auto_trained_teacher.pth")
    if kd_enabled:
        os.makedirs("saved_models", exist_ok=True)
        checkpoint_path = args.teacher_checkpoint or auto_teacher_path
        if args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint):
            checkpoint_found = True
        else:
            checkpoint_found = os.path.exists(checkpoint_path)
        if checkpoint_found:
            teacher = build_teacher(
                num_classes=len(set(BEAT_LABEL_MAP.values())),
                embedding_dim=args.teacher_embedding_dim,
                device=device,
            )
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            teacher.load_state_dict(state)
            print(f"Loaded teacher checkpoint from {checkpoint_path}")
        elif args.auto_train_teacher:
            teacher = build_teacher(
                num_classes=len(set(BEAT_LABEL_MAP.values())),
                embedding_dim=args.teacher_embedding_dim,
                device=device,
            )
            for p in teacher.parameters():
                p.requires_grad = True
            print(
                "[INFO] No teacher checkpoint found. Auto-training a teacher for knowledge distillation."
            )
            train_teacher(
                teacher,
                train_loader,
                val_loader,
                class_weights,
                device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.teacher_auto_train_epochs,
            )
            torch.save({"model_state_dict": teacher.state_dict()}, auto_teacher_path)
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
            print(f"[INFO] Auto-trained teacher saved to {auto_teacher_path}")
        else:
            print("[WARNING] KD enabled but no teacher available. Distillation will be disabled.")
            kd_enabled = False

    if kd_enabled and teacher is not None:
        proj_T = nn.Linear(args.teacher_embedding_dim, args.kd_d).to(device)
        proj_cls = ConstrainedLinear if args.use_value_constraint else nn.Linear
        proj_kwargs = {"scale": args.constraint_scale} if args.use_value_constraint else {}
        proj_S = proj_cls(4, args.kd_d, **proj_kwargs).to(device)
    else:
        proj_T = nn.Identity()
        proj_S = nn.Identity()

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    mse_loss = nn.MSELoss()

    optimizer = Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, verbose=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        student.train()
        running_loss = 0.0
        total = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()

            logits_s, feat_s = student(signals)
            loss_ce = ce_loss_fn(logits_s, labels)
            loss = loss_ce

            if kd_enabled and teacher is not None:
                with torch.no_grad():
                    logits_t, feat_t = teacher(signals)
                kd_logits = kd_logit_loss(logits_s, logits_t, args.kd_temperature)
                f_s = proj_S(scale_to_unit(feat_s) if args.use_value_constraint and not args.use_tanh_activations else feat_s)
                f_t = proj_T(feat_t)
                f_s = l2_normalize(f_s)
                f_t = l2_normalize(f_t)
                kd_feat = mse_loss(f_s, f_t)
                loss = args.alpha * loss_ce + args.beta * kd_logits + args.gamma * kd_feat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)

        val_loss, val_metrics = evaluate(student, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Val F1 {val_metrics['f1']:.3f} Miss {val_metrics['miss_rate'] * 100:.2f}% FPR {val_metrics['fpr'] * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = student.state_dict()
            patience_counter = 0
            print("  -> New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    val_loss, val_metrics = evaluate(student, val_loader, device)
    gen_loss, gen_metrics = evaluate(student, gen_loader, device)
    print(f"Final Val: loss={val_loss:.4f}, F1={val_metrics['f1']:.3f}, miss={val_metrics['miss_rate'] * 100:.2f}%, fpr={val_metrics['fpr'] * 100:.2f}%")
    print(f"Generalization: loss={gen_loss:.4f}, F1={gen_metrics['f1']:.3f}, miss={gen_metrics['miss_rate'] * 100:.2f}%, fpr={gen_metrics['fpr'] * 100:.2f}%")

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "student_model.pth")
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "config": vars(args),
        },
        save_path,
    )
    print(f"Saved student checkpoint to {save_path}")


if __name__ == "__main__":
    main()
