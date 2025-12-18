from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    acc = (y_true == y_pred).mean().item()
    macro_f1 = f1.mean().item()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    pred_counts = np.bincount(y_pred, minlength=4)
    pred_props = pred_counts / max(pred_counts.sum(), 1)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion": cm.tolist(),
        "pred_counts": pred_counts.tolist(),
        "pred_props": pred_props.tolist(),
    }


def format_metrics(tag: str, stats: Dict) -> str:
    lines = [
        f"[{tag}] Acc: {stats['acc']:.4f} MacroF1: {stats['macro_f1']:.4f}",
        f"[{tag}] Precision: {stats['precision']}",
        f"[{tag}] Recall: {stats['recall']}",
        f"[{tag}] F1: {stats['f1']}",
        f"[{tag}] Support: {stats['support']}",
        f"[{tag}] Pred Counts: {stats['pred_counts']} Props: {stats['pred_props']}",
        f"[{tag}] Confusion Matrix:\n{stats['confusion']}",
    ]
    return "\n".join(lines)


def plot_confusion_matrix(cm: np.ndarray, path: str, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["N", "S", "V", "O"], yticklabels=["N", "S", "V", "O"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
