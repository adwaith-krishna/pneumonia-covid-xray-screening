import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Computes comprehensive multi-class classification metrics:
    - Overall Accuracy
    - Macro & Weighted Precision, Recall, F1-Score
    - Per-class metrics
    - Macro ROC-AUC (if predicted probabilities are provided)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))

    # Macro & Weighted averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class breakdown
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(p_class))]

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1": float(f1_class[i]),
            "support": int(support[i]),
        }

    metrics = {
        "accuracy": acc,
        "macro_f1": float(f1_macro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "weighted_f1": float(f1_weighted),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "per_class": per_class,
    }

    # ROC-AUC if probability distribution is supplied
    if y_probs is not None:
        try:
            if len(np.unique(y_true)) == y_probs.shape[1]:
                roc_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
                metrics["macro_roc_auc"] = roc_auc
        except Exception:
            metrics["macro_roc_auc"] = None

    return metrics


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
):
    """
    Plots and saves a high-quality annotated confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth", fontweight="bold", fontsize=11, labelpad=8)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)

    # Annotate values
    thresh = cm_display.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val_str = f"{cm_display[i, j]:{fmt}}"
            if normalize:
                val_str = f"{cm_display[i, j]*100:.1f}%\n({cm[i, j]})"
            ax.text(
                j,
                i,
                val_str,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=9.5,
                fontweight="semibold",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], save_path: str, model_name: str = "Model"):
    """
    Plots training and validation loss, accuracy, and macro F1 history curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#2563EB", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#DC2626", label="Val Loss", linewidth=2)
    ax1.set_title(f"{model_name} - Cross-Entropy Loss", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Macro F1 & Accuracy Curves
    ax2.plot(epochs, history["val_f1"], "o-", color="#059669", label="Val Macro F1", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "^--", color="#D97706", label="Val Accuracy", linewidth=2)
    if "train_f1" in history:
        ax2.plot(epochs, history["train_f1"], ":", color="#6EE7B7", label="Train Macro F1", linewidth=1.5)
    ax2.set_title(f"{model_name} - Validation Macro F1 & Accuracy", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_ylim([0.0, 1.02])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
