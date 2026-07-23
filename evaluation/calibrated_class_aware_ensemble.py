"""Fit calibrated class-aware fusion and export multi-teacher soft targets.

Selection uses a stratified holdout inside VAL.  After selecting the
regularization strength, parameters are refit on all VAL rows and TEST is
evaluated exactly once.  If train probability files are available, fused
train targets are exported for DAMS-TinyPhoBERT distillation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from class_aware_ensemble import (
    fit_class_aware_ensemble,
    fit_temperature,
    validate_expert_probs,
)
from utils.metrics import compute_metrics


console = Console()
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def load_split(model_dirs, split, required=True):
    probs, labels_ref = [], None
    for model_dir in model_dirs:
        prob_path = os.path.join(model_dir, f"{split}_probs.npy")
        label_path = os.path.join(model_dir, f"{split}_labels.npy")
        if not os.path.isfile(prob_path) or not os.path.isfile(label_path):
            if required:
                raise FileNotFoundError(
                    f"Missing {split}_probs.npy or {split}_labels.npy in {model_dir}"
                )
            return None, None
        model_probs = np.load(prob_path)
        model_labels = np.load(label_path).astype(np.int64)
        if labels_ref is None:
            labels_ref = model_labels
        elif not np.array_equal(labels_ref, model_labels):
            raise ValueError(f"Label/order mismatch for split={split}: {model_dir}")
        probs.append(model_probs)
    return validate_expert_probs(np.stack(probs, axis=1)), labels_ref


def metric_dict(labels, probs):
    return compute_metrics(labels.tolist(), probs.argmax(axis=1).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument(
        "--regularization_grid", nargs="+", type=float,
        default=[0.0, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--holdout_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="checkpoints/class_aware_ensemble")
    parser.add_argument(
        "--no_export_train", action="store_true",
        help="Do not export train soft targets even when all experts provide them.",
    )
    parser.add_argument(
        "--train_probs_are_oof", action="store_true",
        help="Declare that every train_probs.npy was generated out-of-fold.",
    )
    args = parser.parse_args()

    names = args.model_names or [Path(path).name for path in args.model_dirs]
    if len(names) != len(args.model_dirs):
        raise ValueError("--model_names must match --model_dirs")
    if not 0.05 <= args.holdout_ratio <= 0.5:
        raise ValueError("--holdout_ratio must be in [0.05, 0.5]")

    val_probs, val_labels = load_split(args.model_dirs, "val")
    test_probs, test_labels = load_split(args.model_dirs, "test")
    if val_probs.shape[1:] != test_probs.shape[1:]:
        raise ValueError("VAL and TEST expert/class dimensions differ")

    fit_idx, holdout_idx = train_test_split(
        np.arange(len(val_labels)),
        test_size=args.holdout_ratio,
        random_state=args.seed,
        stratify=val_labels,
    )
    fit_probs, fit_labels = val_probs[fit_idx], val_labels[fit_idx]
    holdout_probs, holdout_labels = val_probs[holdout_idx], val_labels[holdout_idx]

    # Fit temperatures on gate-fit only while selecting regularization.
    selection_temperatures = np.array(
        [
            fit_temperature(fit_probs[:, m], fit_labels)
            for m in range(fit_probs.shape[1])
        ]
    )
    candidates = []
    best_lambda, best_score = None, -np.inf
    for regularization in args.regularization_grid:
        model = fit_class_aware_ensemble(
            fit_probs, fit_labels,
            temperatures=selection_temperatures,
            regularization=regularization,
        )
        metrics = metric_dict(holdout_labels, model.predict_proba(holdout_probs))
        candidates.append({"regularization": regularization, **metrics})
        if metrics["macro_f1"] > best_score:
            best_lambda, best_score = regularization, metrics["macro_f1"]

    # Refit all calibration/fusion parameters on all validation data.
    final_temperatures = np.array(
        [
            fit_temperature(val_probs[:, m], val_labels)
            for m in range(val_probs.shape[1])
        ]
    )
    model = fit_class_aware_ensemble(
        val_probs, val_labels,
        temperatures=final_temperatures,
        regularization=best_lambda,
    )
    val_fused = model.predict_proba(val_probs)
    test_fused = model.predict_proba(test_probs)
    val_metrics = metric_dict(val_labels, val_fused)
    test_metrics = metric_dict(test_labels, test_fused)
    individual_test = {
        name: metric_dict(test_labels, test_probs[:, idx])
        for idx, name in enumerate(names)
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "val_probs.npy", val_fused.astype(np.float32))
    np.save(output_dir / "val_labels.npy", val_labels)
    np.save(output_dir / "test_probs.npy", test_fused.astype(np.float32))
    np.save(output_dir / "test_labels.npy", test_labels)
    exported_train = False
    train_probs, train_labels = load_split(
        args.model_dirs, "train", required=False
    )
    if train_probs is not None and not args.no_export_train:
        train_fused = model.predict_proba(train_probs)
        np.save(output_dir / "train_probs.npy", train_fused.astype(np.float32))
        np.save(output_dir / "train_labels.npy", train_labels)
        exported_train = True
        console.print(
            "[yellow]Train targets exported. They are in-sample unless the "
            "expert files were produced by OOF inference; use OOF for paper claims.[/yellow]"
        )

    metadata = {
        "method": "calibrated_class_aware_log_opinion_pool",
        "model_names": names,
        "model_dirs": args.model_dirs,
        "selection": {
            "seed": args.seed,
            "holdout_ratio": args.holdout_ratio,
            "candidates": candidates,
            "selected_regularization": best_lambda,
            "holdout_macro_f1": best_score,
        },
        "parameters": model.to_dict(),
        "val_metrics_refit": val_metrics,
        "test_metrics": test_metrics,
        "individual_test_metrics": individual_test,
        "train_targets_exported": exported_train,
        "train_targets_oof": bool(exported_train and args.train_probs_are_oof),
    }
    with open(output_dir / "ensemble_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    table = Table(title="Calibrated Class-Aware Ensemble")
    table.add_column("Metric")
    table.add_column("VAL (refit)")
    table.add_column("TEST")
    for key in ["accuracy", "macro_f1", "f1_clean", "f1_offensive", "f1_hate"]:
        table.add_row(key, f"{val_metrics[key]:.4f}", f"{test_metrics[key]:.4f}")
    console.print(table)
    console.print("Temperatures: " + ", ".join(
        f"{name}={temp:.3f}" for name, temp in zip(names, final_temperatures)
    ))
    for class_idx, class_name in enumerate(LABEL_NAMES):
        weights = ", ".join(
            f"{name}={model.class_weights[m, class_idx]:.3f}"
            for m, name in enumerate(names)
        )
        console.print(f"{class_name}: {weights}")
    console.print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
