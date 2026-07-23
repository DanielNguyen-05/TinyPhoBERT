"""Export a pre-registered fixed weighted ensemble for distillation.

Unlike the class-aware meta-learner, this script performs no fitting and no
calibration.  It is useful for reproducing a weight vector that was selected
on validation before the test set was inspected.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.calibrated_class_aware_ensemble import load_split, metric_dict


console = Console()


def combine(expert_probs, weights):
    probabilities = (
        expert_probs * weights[None, :, None]
    ).sum(axis=1)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument("--weights", nargs="+", type=float, required=True)
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
    weights = np.asarray(args.weights, dtype=np.float64)
    if weights.shape != (len(args.model_dirs),):
        raise ValueError("--weights must contain one value per model")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("--weights must be non-negative with a positive sum")
    weights /= weights.sum()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = {}
    exported_splits = []
    for split in ["train", "val", "test"]:
        expert_probs, labels, sample_ids = load_split(
            args.model_dirs, split, required=(split != "train")
        )
        if expert_probs is None:
            continue
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
        "method": "fixed_arithmetic_probability_pool",
        "model_names": names,
        "model_dirs": args.model_dirs,
        "weights": dict(zip(names, weights.tolist())),
        "split_metrics": split_metrics,
        "exported_splits": exported_splits,
        "train_targets_oof": bool(
            "train" in exported_splits and args.train_probs_are_oof
        ),
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
    console.print(table)
    console.print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
