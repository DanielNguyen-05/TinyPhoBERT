"""Tune class decision biases on validation macro-F1, then apply once to test."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compute_metrics


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, 1e-12, 1.0)) + bias


def scores_to_probs(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--min_bias", type=float, default=-1.0)
    parser.add_argument("--max_bias", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for calibrated train/val/test distillation targets.",
    )
    args = parser.parse_args()

    val_probs = np.load(os.path.join(args.model_dir, "val_probs.npy"))
    val_labels = np.load(os.path.join(args.model_dir, "val_labels.npy"))
    test_probs = np.load(os.path.join(args.model_dir, "test_probs.npy"))
    test_labels = np.load(os.path.join(args.model_dir, "test_labels.npy"))

    values = np.arange(args.min_bias, args.max_bias + args.step / 2, args.step)
    best_bias = np.zeros(3)
    best_score = -1.0
    # CLEAN is the reference class. Search OFFENSIVE and HATE thresholds.
    for offensive_bias in values:
        for hate_bias in values:
            bias = np.array([0.0, offensive_bias, hate_bias])
            preds = apply_bias(val_probs, bias).argmax(axis=1)
            score = compute_metrics(val_labels.tolist(), preds.tolist())["macro_f1"]
            if score > best_score:
                best_score, best_bias = score, bias.copy()

    test_logits = apply_bias(test_probs, best_bias)
    test_preds = test_logits.argmax(axis=1)
    metrics = compute_metrics(test_labels.tolist(), test_preds.tolist())
    np.save(os.path.join(args.model_dir, "test_scores_class_bias.npy"), test_logits)

    output_dir = args.output_dir or os.path.join(
        args.model_dir, "class_bias_calibrated"
    )
    os.makedirs(output_dir, exist_ok=True)
    source_metadata_path = os.path.join(
        args.model_dir, "ensemble_metadata.json"
    )
    source_train_targets_oof = False
    if os.path.isfile(source_metadata_path):
        with open(source_metadata_path) as source_file:
            source_metadata = json.load(source_file)
        source_train_targets_oof = source_metadata.get(
            "train_targets_oof", False
        )
    exported_splits = []
    for split in ["train", "val", "test"]:
        probs_path = os.path.join(args.model_dir, f"{split}_probs.npy")
        labels_path = os.path.join(args.model_dir, f"{split}_labels.npy")
        if not os.path.isfile(probs_path) or not os.path.isfile(labels_path):
            continue
        split_probs = np.load(probs_path)
        split_labels = np.load(labels_path)
        calibrated_probs = scores_to_probs(
            apply_bias(split_probs, best_bias)
        )
        np.save(
            os.path.join(output_dir, f"{split}_probs.npy"),
            calibrated_probs.astype(np.float32),
        )
        np.save(
            os.path.join(output_dir, f"{split}_labels.npy"), split_labels
        )
        ids_path = os.path.join(args.model_dir, f"{split}_sample_ids.npy")
        if os.path.isfile(ids_path):
            np.save(
                os.path.join(output_dir, f"{split}_sample_ids.npy"),
                np.load(ids_path),
            )
        exported_splits.append(split)

    payload = {
        "method": "validation_selected_class_logit_bias",
        "source_model_dir": args.model_dir,
        "validation_macro_f1": float(best_score),
        "class_bias": best_bias.tolist(),
        "test_metrics": metrics,
        "exported_splits": exported_splits,
        "train_targets_oof": source_train_targets_oof,
    }
    output_path = os.path.join(output_dir, "ensemble_metadata.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Calibrated targets saved to {output_dir}")


if __name__ == "__main__":
    main()
