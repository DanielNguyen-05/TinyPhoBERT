"""
utils/metrics.py
Evaluation metrics for hate speech detection.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for hate speech detection.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        label_names: Class names for reporting.

    Returns:
        Dictionary with accuracy, macro P/R/F1, and per-class F1.
    """
    if label_names is None:
        label_names = LABEL_NAMES

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    for i, name in enumerate(label_names):
        if i < len(per_class_f1):
            metrics[f"f1_{name.lower()}"] = per_class_f1[i]

    return metrics


def compute_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute metrics from raw logits."""
    preds = np.argmax(logits, axis=-1)
    return compute_metrics(labels.tolist(), preds.tolist(), label_names)


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> None:
    """Print a detailed classification report."""
    if label_names is None:
        label_names = LABEL_NAMES
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    print(report)


def get_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
) -> np.ndarray:
    """Return the confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def huggingface_compute_metrics(eval_pred):
    """
    HuggingFace Trainer-compatible compute_metrics function.
    Expects eval_pred to be an EvalPrediction namedtuple with
    .predictions (logits) and .label_ids (true labels).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    # Rename for Trainer compatibility
    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["macro_f1"],
        "precision_macro": metrics["macro_precision"],
        "recall_macro": metrics["macro_recall"],
    }
