import argparse
import os
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BeatDataset, load_datasets
from metrics import compute_classification_metrics, format_metrics, plot_confusion_matrix
from model import SegmentAwareCNN
from sampler import build_weighted_sampler, effective_num_weights, run_sampler_sanity_check
from utils import save_json, set_seed


def collate_fn(batch):
    beats, labels = zip(*batch)
    x = torch.tensor(np.stack(beats), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def build_loaders(train_ds: BeatDataset, val_ds: BeatDataset, gen_ds: BeatDataset, batch_size: int, target_mix=None):
    sampler, train_counts = build_weighted_sampler(train_ds.y, target_mix)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    gen_loader = DataLoader(gen_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if len(gen_ds) > 0 else None
    return train_loader, val_loader, gen_loader, train_counts


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, prior_log: torch.Tensor, tau: float, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        logits = logits - tau * prior_log
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    stats = compute_classification_metrics(y_true, y_pred)
    return total_loss / len(loader.dataset), stats


def maybe_save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, stats: Dict):
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "stats": stats}, path)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Segment-aware ECG four-class training")
    parser.add_argument(
        "--data_root",
        type=str,
        default="E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/",
        help="Root directory of MIT-BIH records",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio (by records) from TRAIN_RECORDS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--normalization", type=str, default="zscore", choices=["zscore", "robust"], help="Beat normalization type")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=1.0, help="Logit adjustment temperature")
    parser.add_argument("--beta", type=float, default=0.9999, help="Effective number beta")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store logs, checkpoints, and plots")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_output_dir(args.output_dir)
    train_ds, val_ds, gen_ds = load_datasets(args.data_root, args.val_ratio, args.seed, args.normalization)
    target_mix = {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}
    train_loader, val_loader, gen_loader, train_counts = build_loaders(train_ds, val_ds, gen_ds, args.batch_size, target_mix)
    run_sampler_sanity_check(train_loader, num_batches=20)

    for cls, cnt in train_counts.items():
        if cnt == 0:
            raise RuntimeError(
                f"Class {cls} has zero samples in the training split. Adjust seed/val_ratio so each class is present before training."
            )

    prior = torch.tensor([train_counts[i] for i in range(4)], dtype=torch.float32)
    prior = prior / prior.sum()
    prior_log = torch.log(prior + 1e-12).to(device)

    class_weights = effective_num_weights(train_counts, beta=args.beta).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SegmentAwareCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = {}
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, prior_log, args.tau, args.grad_clip)
        val_loss, val_stats = evaluate(model, val_loader, criterion, device)
        gen_loss, gen_stats = (None, None)
        if gen_loader is not None:
            gen_loss, gen_stats = evaluate(model, gen_loader, criterion, device)

        elapsed = time.time() - start_time
        header = f"Epoch {epoch}: TrainLoss {train_loss:.4f} ValLoss {val_loss:.4f} ValAcc {val_stats['acc']:.4f} ValMacroF1 {val_stats['macro_f1']:.4f} Time {elapsed:.1f}s"
        print(header)
        log_lines.append(header)
        val_block = format_metrics("Val", val_stats)
        print(val_block)
        log_lines.append(val_block)
        if gen_stats is not None:
            gen_loss_line = f"[Gen] Loss: {gen_loss:.4f}"
            print(gen_loss_line)
            log_lines.append(gen_loss_line)
            gen_block = format_metrics("Gen", gen_stats)
            print(gen_block)
            log_lines.append(gen_block)

        history[epoch] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val": val_stats,
            "gen_loss": gen_loss,
            "gen": gen_stats,
        }

        if val_stats["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_stats["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            maybe_save_checkpoint(os.path.join(args.output_dir, "best.pt"), model, optimizer, epoch, {"val": val_stats, "gen": gen_stats})
        else:
            epochs_no_improve += 1

        maybe_save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch, {"val": val_stats, "gen": gen_stats})

        if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
            print("Early stopping triggered")
            break

    results = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "hyperparams": vars(args),
        "history": history,
    }
    if gen_loader is not None and history.get(best_epoch, {}).get("gen"):
        results["best_gen"] = history[best_epoch]["gen"]
    results_path = os.path.join(args.output_dir, "results.json")
    save_json(results, results_path)
    log_path = os.path.join(args.output_dir, "train.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Visualization of tracked metrics
    epochs = sorted(history.keys())
    train_losses = [history[e]["train_loss"] for e in epochs]
    val_losses = [history[e]["val_loss"] for e in epochs]
    val_macro_f1s = [history[e]["val"]["macro_f1"] for e in epochs]
    val_accs = [history[e]["val"]["acc"] for e in epochs]
    gen_macro_f1s = [history[e]["gen"]["macro_f1"] if history[e]["gen"] else None for e in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.plot(epochs, val_macro_f1s, label="Val Macro F1")
    plt.plot(epochs, val_accs, label="Val Acc")
    if any(g is not None for g in gen_macro_f1s):
        plt.plot([e for e, g in zip(epochs, gen_macro_f1s) if g is not None], [g for g in gen_macro_f1s if g is not None], label="Gen Macro F1", linestyle="--")
    plt.xlabel("Epoch")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "metrics_tracking.png"))
    plt.close()

    best_val = history[best_epoch]["val"] if best_epoch in history else None
    if best_val:
        plot_confusion_matrix(np.array(best_val["confusion"]), os.path.join(args.output_dir, "confusion_val.png"), title=f"Val Confusion (epoch {best_epoch})")
    best_gen = history[best_epoch].get("gen") if best_epoch in history else None
    if best_gen:
        plot_confusion_matrix(np.array(best_gen["confusion"]), os.path.join(args.output_dir, "confusion_gen.png"), title=f"Gen Confusion (epoch {best_epoch})")

    print(f"Training complete. Best epoch {best_epoch} Val MacroF1 {best_macro_f1:.4f}")
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
