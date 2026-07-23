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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--min_bias", type=float, default=-1.0)
    parser.add_argument("--max_bias", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
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

    payload = {
        "validation_macro_f1": float(best_score),
        "class_bias": best_bias.tolist(),
        "test_metrics": metrics,
    }
    output_path = os.path.join(args.model_dir, "class_bias_results.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
