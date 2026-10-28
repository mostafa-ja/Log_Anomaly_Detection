"""
Utility functions for Log Anomaly Detection Suite:
- Seed management and reproducibility
- Device selection (CUDA / CPU)
- Metric calculation (Precision, Recall, F1, Accuracy, ROC-AUC)
- Publication-quality visualization helpers
"""

import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc,
)


def seed_everything(seed: int = 42) -> None:
    """Set seeds across random, numpy, and torch for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and return available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def compute_metrics(y_true, y_pred, y_scores=None) -> dict:
    """
    Calculate classification metrics:
    TP, FP, TN, FN, Precision, Recall, F1-Score, Accuracy, and ROC-AUC (if scores provided).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    roc_auc = None
    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except Exception:
            roc_auc = 0.5

    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "ROC_AUC": float(roc_auc) if roc_auc is not None else None,
    }
    return metrics


def plot_confusion_matrix(cm, labels=["Normal", "Anomaly"], title="Confusion Matrix", save_path=None):
    """Plot an aesthetic confusion matrix heatmap using Seaborn."""
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("True Label", fontsize=12, fontweight="medium")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="medium")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_scores, title="Receiver Operating Characteristic (ROC)", save_path=None):
    """Plot aesthetic ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
