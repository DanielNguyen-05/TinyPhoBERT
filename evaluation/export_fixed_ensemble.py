"""Export a pre-registered fixed weighted ensemble for distillation.

Unlike the class-aware meta-learner, this script performs no fitting and no
calibration.  It is useful for reproducing a weight vector that was selected
on validation before the test set was inspected.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.calibrated_class_aware_ensemble import (
    fit_static_macro_f1,
    load_split,
    metric_dict,
)
from class_aware_ensemble import fit_temperature, temperature_scale_probs


console = Console()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine(expert_probs, weights):
    probabilities = (
        expert_probs * weights[None, :, None]
    ).sum(axis=1)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def calibrate_experts(expert_probs, temperatures, enabled):
    if not enabled:
        return expert_probs
    return np.stack([
        temperature_scale_probs(
            expert_probs[:, model_idx], temperatures[model_idx]
        )
        for model_idx in range(expert_probs.shape[1])
    ], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+")
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument("--weights", nargs="+", type=float)
    weight_group.add_argument(
        "--search_weights", action="store_true",
        help="Grid-search simplex weights on VAL macro-F1.",
    )
    parser.add_argument("--weight_step", type=float, default=0.1)
    parser.add_argument(
        "--temperature_mode",
        choices=["none", "grid", "continuous"],
        default="none",
        help="Fit temperatures on VAL before applying the fixed weights.",
    )
    parser.add_argument(
        "--output_dir", default="checkpoints/fixed_weight_ensemble"
    )
    parser.add_argument(
        "--train_probs_are_oof", action="store_true",
        help="Set only when every expert train prediction is genuinely OOF.",
    )
    args = parser.parse_args()

    names = args.model_names or [
        Path(model_dir).name for model_dir in args.model_dirs
    ]
    if len(names) != len(args.model_dirs):
        raise ValueError("--model_names must match --model_dirs")
    loaded_splits = {}
    for split in ["train", "val", "test"]:
        loaded_splits[split] = load_split(
            args.model_dirs, split, required=(split != "train")
        )

    val_expert_probs, val_labels, _ = loaded_splits["val"]
    temperatures = np.ones(len(args.model_dirs), dtype=np.float64)
    if args.temperature_mode == "continuous":
        temperatures = np.asarray([
            fit_temperature(val_expert_probs[:, model_idx], val_labels)
            for model_idx in range(val_expert_probs.shape[1])
        ])
    elif args.temperature_mode == "grid":
        grid = np.arange(0.5, 3.01, 0.05)
        for model_idx in range(val_expert_probs.shape[1]):
            best_temperature, best_nll = 1.0, np.inf
            for temperature in grid:
                calibrated = temperature_scale_probs(
                    val_expert_probs[:, model_idx], float(temperature)
                )
                nll = -np.log(np.clip(
                    calibrated[np.arange(len(val_labels)), val_labels],
                    1e-12, 1.0,
                )).mean()
                if nll < best_nll:
                    best_temperature = float(temperature)
                    best_nll = float(nll)
            temperatures[model_idx] = best_temperature

    calibrated_val = calibrate_experts(
        val_expert_probs,
        temperatures,
        enabled=args.temperature_mode != "none",
    )
    if args.search_weights:
        weights, searched_val_score = fit_static_macro_f1(
            calibrated_val, val_labels, args.weight_step
        )
    else:
        weights = np.asarray(args.weights, dtype=np.float64)
        if weights.shape != (len(args.model_dirs),):
            raise ValueError("--weights must contain one value per model")
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                "--weights must be non-negative with a positive sum"
            )
        weights /= weights.sum()
        searched_val_score = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = {}
    exported_splits = []
    for split in ["train", "val", "test"]:
        expert_probs, labels, sample_ids = loaded_splits[split]
        if expert_probs is None:
            continue
        expert_probs = calibrate_experts(
            expert_probs,
            temperatures,
            enabled=args.temperature_mode != "none",
        )
        fused_probs = combine(expert_probs, weights)
        np.save(
            output_dir / f"{split}_probs.npy",
            fused_probs.astype(np.float32),
        )
        np.save(output_dir / f"{split}_labels.npy", labels)
        if sample_ids is not None:
            np.save(output_dir / f"{split}_sample_ids.npy", sample_ids)
        split_metrics[split] = metric_dict(labels, fused_probs)
        exported_splits.append(split)

    metadata = {
        "method": (
            "val_grid_searched_arithmetic_probability_pool"
            if args.search_weights
            else "fixed_arithmetic_probability_pool"
        ),
        "model_names": names,
        "model_dirs": args.model_dirs,
        "weights": dict(zip(names, weights.tolist())),
        "temperature_mode": args.temperature_mode,
        "temperatures": dict(zip(names, temperatures.tolist())),
        "weight_step": args.weight_step if args.search_weights else None,
        "searched_val_macro_f1": searched_val_score,
        "split_metrics": split_metrics,
        "exported_splits": exported_splits,
        "train_targets_oof": bool(
            "train" in exported_splits and args.train_probs_are_oof
        ),
        "source_fingerprints": {
            name: {
                filename: sha256_file(Path(model_dir) / filename)
                for filename in [
                    "val_probs.npy", "val_labels.npy",
                    "test_probs.npy", "test_labels.npy",
                ]
            }
            for name, model_dir in zip(names, args.model_dirs)
        },
    }
    with open(output_dir / "ensemble_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    table = Table(title="Fixed Weighted Ensemble")
    table.add_column("Metric")
    table.add_column("VAL")
    table.add_column("TEST")
    for key in [
        "accuracy", "macro_f1", "f1_clean", "f1_offensive", "f1_hate"
    ]:
        table.add_row(
            key,
            f"{split_metrics['val'][key]:.4f}",
            f"{split_metrics['test'][key]:.4f}",
        )
    console.print(
        "Weights: " + ", ".join(
            f"{name}={weight:.2f}"
            for name, weight in zip(names, weights)
        )
    )
    console.print(
        "Temperatures: " + ", ".join(
            f"{name}={temperature:.2f}"
            for name, temperature in zip(names, temperatures)
        )
    )
    console.print(table)
    console.print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
