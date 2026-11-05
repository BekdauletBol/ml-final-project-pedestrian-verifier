"""Metrics and plots: accuracy, precision, recall, F1, confusion matrix, curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .data import CLASSES


def metrics_dict(y_true, y_pred) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pp, rp, fp, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0], average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "precision_pedestrian": float(pp[0]),
        "recall_pedestrian": float(rp[0]),
        "f1_pedestrian": float(fp[0]),
    }


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)


def plot_confusion(y_true, y_pred, title: str, path: str | Path | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, fmt, sub in [(axes[0], cm, "d", "counts"), (axes[1], cm_norm, ".2f", "row-normalized")]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASSES, yticklabels=CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} ({sub})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_history(histories: dict, path: str | Path | None = None):
    """Loss and accuracy curves (train dashed, validation solid) for one or more runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (name, h) in enumerate(histories.items()):
        color = f"C{i}"
        ep = np.arange(1, len(h.train_loss) + 1)
        axes[0].plot(ep, h.train_loss, "--", color=color, alpha=0.7, label=f"{name} train")
        axes[0].plot(ep, h.val_loss, "-", color=color, label=f"{name} val")
        axes[1].plot(ep, h.train_acc, "--", color=color, alpha=0.7, label=f"{name} train")
        axes[1].plot(ep, h.val_acc, "-", color=color, label=f"{name} val")
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
