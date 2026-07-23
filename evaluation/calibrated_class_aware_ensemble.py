"""Fit calibrated class-aware fusion and export multi-teacher soft targets.

Selection uses repeated stratified cross-validation inside VAL.  Parameters
are then refit on all VAL rows and TEST is evaluated exactly once.  If train
probability files are available, fused train targets are exported for
DAMS-TinyPhoBERT distillation.
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from class_aware_ensemble import (
    fit_class_aware_ensemble,
    fit_temperature,
    temperature_scale_probs,
    validate_expert_probs,
)
from utils.metrics import compute_metrics


console = Console()
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def load_split(model_dirs, split, required=True):
    loaded = []
    for model_dir in model_dirs:
        prob_path = os.path.join(model_dir, f"{split}_probs.npy")
        label_path = os.path.join(model_dir, f"{split}_labels.npy")
        if not os.path.isfile(prob_path) or not os.path.isfile(label_path):
            if required:
                raise FileNotFoundError(
                    f"Missing {split}_probs.npy or {split}_labels.npy in {model_dir}"
                )
            return None, None, None
        model_probs = np.load(prob_path)
        model_labels = np.load(label_path).astype(np.int64)
        id_path = os.path.join(model_dir, f"{split}_sample_ids.npy")
        sample_ids = (
            np.load(id_path).astype(str) if os.path.isfile(id_path) else None
        )
        if len(model_probs) != len(model_labels):
            raise ValueError(f"Probability/label length mismatch: {model_dir}")
        if sample_ids is not None and len(sample_ids) != len(model_labels):
            raise ValueError(f"sample_id length mismatch: {model_dir}")
        loaded.append([model_probs, model_labels, sample_ids, model_dir])

    ids_available = sum(item[2] is not None for item in loaded)
    reference_ids = None
    if ids_available == len(loaded):
        reference_ids = loaded[0][2]
        if len(set(reference_ids.tolist())) != len(reference_ids):
            raise ValueError(f"Duplicate sample IDs in {loaded[0][3]}/{split}")
        reference_set = set(reference_ids.tolist())
        for item in loaded[1:]:
            sample_ids = item[2]
            if len(set(sample_ids.tolist())) != len(sample_ids):
                raise ValueError(f"Duplicate sample IDs in {item[3]}/{split}")
            if set(sample_ids.tolist()) != reference_set:
                raise ValueError(
                    f"Different sample ID sets for split={split}: {item[3]}"
                )
            row_by_id = {
                sample_id: row_idx
                for row_idx, sample_id in enumerate(sample_ids.tolist())
            }
            order = np.asarray(
                [row_by_id[sample_id] for sample_id in reference_ids.tolist()]
            )
            item[0], item[1], item[2] = (
                item[0][order], item[1][order], item[2][order]
            )
    elif ids_available:
        console.print(
            f"[yellow]{split}: only {ids_available}/{len(loaded)} experts "
            "have sample IDs; falling back to positional checks. Re-export all "
            "experts for strict alignment.[/yellow]"
        )

    labels_ref = loaded[0][1]
    for _, model_labels, _, model_dir in loaded[1:]:
        if not np.array_equal(labels_ref, model_labels):
            raise ValueError(f"Label/order mismatch for split={split}: {model_dir}")
    probs = [item[0] for item in loaded]
    return (
        validate_expert_probs(np.stack(probs, axis=1)),
        labels_ref,
        reference_ids,
    )


def metric_dict(labels, probs):
    return compute_metrics(labels.tolist(), probs.argmax(axis=1).tolist())


def simplex_weights(n_experts, step):
    n_steps = int(round(1.0 / step))
    if not np.isclose(n_steps * step, 1.0):
        raise ValueError("--weight_step must evenly divide 1.0")
    for prefix in itertools.product(range(n_steps + 1), repeat=n_experts - 1):
        if sum(prefix) <= n_steps:
            yield np.asarray(
                (*prefix, n_steps - sum(prefix)), dtype=np.float64
            ) / n_steps


def fit_static_macro_f1(expert_probs, labels, step):
    best_weights, best_score = None, -np.inf
    for weights in simplex_weights(expert_probs.shape[1], step):
        combined = (expert_probs * weights[None, :, None]).sum(axis=1)
        score = metric_dict(labels, combined)["macro_f1"]
        if score > best_score:
            best_weights, best_score = weights, score
    return best_weights, best_score


def static_predict(expert_probs, weights):
    probs = (expert_probs * weights[None, :, None]).sum(axis=1)
    return probs / probs.sum(axis=1, keepdims=True)


def calibrate_stack(expert_probs, temperatures):
    return np.stack(
        [
            temperature_scale_probs(expert_probs[:, model_idx], temperature)
            for model_idx, temperature in enumerate(temperatures)
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument(
        "--regularization_grid", nargs="+", type=float,
        default=[0.0, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--class_balance_power_grid", nargs="+", type=float,
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "--blend_grid", nargs="+", type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="0=strong static ensemble, 1=fully class-aware.",
    )
    parser.add_argument("--weight_step", type=float, default=0.1)
    parser.add_argument("--selection_folds", type=int, default=3)
    parser.add_argument("--selection_repeats", type=int, default=2)
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
    if args.selection_folds < 2 or args.selection_repeats < 1:
        raise ValueError("selection_folds >= 2 and selection_repeats >= 1 required")
    if any(not 0.0 <= value <= 1.0 for value in args.class_balance_power_grid):
        raise ValueError("--class_balance_power_grid values must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in args.blend_grid):
        raise ValueError("--blend_grid values must be in [0, 1]")

    val_probs, val_labels, val_ids = load_split(args.model_dirs, "val")
    test_probs, test_labels, test_ids = load_split(args.model_dirs, "test")
    if val_probs.shape[1:] != test_probs.shape[1:]:
        raise ValueError("VAL and TEST expert/class dimensions differ")

    splitter = RepeatedStratifiedKFold(
        n_splits=args.selection_folds,
        n_repeats=args.selection_repeats,
        random_state=args.seed,
    )
    candidate_scores = {
        (regularization, balance_power, blend): []
        for regularization in args.regularization_grid
        for balance_power in args.class_balance_power_grid
        for blend in args.blend_grid
    }
    raw_fold_scores, calibrated_fold_scores = [], []
    for fold_idx, (fit_idx, holdout_idx) in enumerate(
        splitter.split(val_probs, val_labels), start=1
    ):
        fit_probs, fit_labels = val_probs[fit_idx], val_labels[fit_idx]
        holdout_probs = val_probs[holdout_idx]
        holdout_labels = val_labels[holdout_idx]
        fold_temperatures = np.array(
            [
                fit_temperature(fit_probs[:, model_idx], fit_labels)
                for model_idx in range(fit_probs.shape[1])
            ]
        )
        fit_calibrated = calibrate_stack(fit_probs, fold_temperatures)
        raw_anchor, raw_fit_score = fit_static_macro_f1(
            fit_probs, fit_labels, args.weight_step
        )
        calibrated_anchor, calibrated_fit_score = fit_static_macro_f1(
            fit_calibrated, fit_labels, args.weight_step
        )
        raw_fold_scores.append(raw_fit_score)
        calibrated_fold_scores.append(calibrated_fit_score)
        raw_holdout = static_predict(holdout_probs, raw_anchor)
        for regularization in args.regularization_grid:
            for balance_power in args.class_balance_power_grid:
                candidate_model = fit_class_aware_ensemble(
                    fit_probs, fit_labels,
                    temperatures=fold_temperatures,
                    regularization=regularization,
                    class_balance_power=balance_power,
                    anchor_weights=calibrated_anchor,
                    pooling="linear",
                )
                class_aware_holdout = candidate_model.predict_proba(
                    holdout_probs
                )
                for blend in args.blend_grid:
                    blended = (
                        blend * class_aware_holdout
                        + (1.0 - blend) * raw_holdout
                    )
                    candidate_scores[
                        (regularization, balance_power, blend)
                    ].append(
                        metric_dict(holdout_labels, blended)["macro_f1"]
                    )
        console.print(
            f"  selection fold {fold_idx}/"
            f"{args.selection_folds * args.selection_repeats} complete"
        )

    candidates = []
    for (regularization, balance_power, blend), scores in candidate_scores.items():
        candidates.append({
            "regularization": regularization,
            "class_balance_power": balance_power,
            "class_aware_blend": blend,
            "cv_macro_f1_mean": float(np.mean(scores)),
            "cv_macro_f1_std": float(np.std(scores)),
            "fold_scores": [float(score) for score in scores],
        })
    best_config = max(
        candidates,
        key=lambda item: (
            item["cv_macro_f1_mean"],
            -item["cv_macro_f1_std"],
        ),
    )
    best_score = best_config["cv_macro_f1_mean"]

    # Refit all calibration/fusion parameters on all validation data.
    final_temperatures = np.array(
        [
            fit_temperature(val_probs[:, m], val_labels)
            for m in range(val_probs.shape[1])
        ]
    )
    val_calibrated = calibrate_stack(val_probs, final_temperatures)
    raw_anchor, raw_val_score = fit_static_macro_f1(
        val_probs, val_labels, args.weight_step
    )
    calibrated_anchor, calibrated_val_score = fit_static_macro_f1(
        val_calibrated, val_labels, args.weight_step
    )
    model = fit_class_aware_ensemble(
        val_probs, val_labels,
        temperatures=final_temperatures,
        regularization=best_config["regularization"],
        class_balance_power=best_config["class_balance_power"],
        anchor_weights=calibrated_anchor,
        pooling="linear",
    )
    blend = best_config["class_aware_blend"]
    val_class_aware = model.predict_proba(val_probs)
    test_class_aware = model.predict_proba(test_probs)
    val_static = static_predict(val_probs, raw_anchor)
    test_static = static_predict(test_probs, raw_anchor)
    val_fused = blend * val_class_aware + (1.0 - blend) * val_static
    test_fused = blend * test_class_aware + (1.0 - blend) * test_static
    val_metrics = metric_dict(val_labels, val_fused)
    test_metrics = metric_dict(test_labels, test_fused)
    static_test_metrics = metric_dict(test_labels, test_static)
    class_aware_test_metrics = metric_dict(test_labels, test_class_aware)
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
    if val_ids is not None:
        np.save(output_dir / "val_sample_ids.npy", val_ids)
    if test_ids is not None:
        np.save(output_dir / "test_sample_ids.npy", test_ids)
    exported_train = False
    train_probs, train_labels, train_ids = load_split(
        args.model_dirs, "train", required=False
    )
    if train_probs is not None and not args.no_export_train:
        train_class_aware = model.predict_proba(train_probs)
        train_static = static_predict(train_probs, raw_anchor)
        train_fused = blend * train_class_aware + (1.0 - blend) * train_static
        np.save(output_dir / "train_probs.npy", train_fused.astype(np.float32))
        np.save(output_dir / "train_labels.npy", train_labels)
        if train_ids is not None:
            np.save(output_dir / "train_sample_ids.npy", train_ids)
        exported_train = True
        console.print(
            "[yellow]Train targets exported. They are in-sample unless the "
            "expert files were produced by OOF inference. This is valid for "
            "standard KD; use OOF for leakage-free stacking claims and to "
            "reduce train/evaluation confidence shift.[/yellow]"
        )

    metadata = {
        "method": "anchored_calibrated_class_aware_linear_pool",
        "model_names": names,
        "model_dirs": args.model_dirs,
        "selection": {
            "seed": args.seed,
            "folds": args.selection_folds,
            "repeats": args.selection_repeats,
            "mean_raw_static_fit_macro_f1": float(
                np.mean(raw_fold_scores)
            ),
            "mean_calibrated_static_fit_macro_f1": float(
                np.mean(calibrated_fold_scores)
            ),
            "candidates": candidates,
            "selected_config": {
                key: best_config[key]
                for key in [
                    "regularization", "class_balance_power",
                    "class_aware_blend",
                ]
            },
            "selected_cv_macro_f1": best_score,
        },
        "static_raw_weights": dict(zip(names, raw_anchor.tolist())),
        "static_raw_val_macro_f1": raw_val_score,
        "static_calibrated_weights": dict(
            zip(names, calibrated_anchor.tolist())
        ),
        "static_calibrated_val_macro_f1": calibrated_val_score,
        "parameters": model.to_dict(),
        "val_metrics_refit": val_metrics,
        "test_metrics": test_metrics,
        "static_test_metrics": static_test_metrics,
        "unblended_class_aware_test_metrics": class_aware_test_metrics,
        "individual_test_metrics": individual_test,
        "train_targets_exported": exported_train,
        "train_targets_oof": bool(exported_train and args.train_probs_are_oof),
    }
    with open(output_dir / "ensemble_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    table = Table(title="Calibrated Class-Aware Ensemble")
    table.add_column("Metric")
    table.add_column("VAL selected")
    table.add_column("TEST static")
    table.add_column("TEST class-aware")
    table.add_column("TEST selected")
    for key in ["accuracy", "macro_f1", "f1_clean", "f1_offensive", "f1_hate"]:
        table.add_row(
            key,
            f"{val_metrics[key]:.4f}",
            f"{static_test_metrics[key]:.4f}",
            f"{class_aware_test_metrics[key]:.4f}",
            f"{test_metrics[key]:.4f}",
        )
    console.print(table)
    console.print("Temperatures: " + ", ".join(
        f"{name}={temp:.3f}" for name, temp in zip(names, final_temperatures)
    ))
    console.print(
        "Selected: "
        f"lambda={best_config['regularization']}, "
        f"balance_power={best_config['class_balance_power']}, "
        f"class_aware_blend={blend}"
    )
    for class_idx, class_name in enumerate(LABEL_NAMES):
        weights = ", ".join(
            f"{name}={model.class_weights[m, class_idx]:.3f}"
            for m, name in enumerate(names)
        )
        console.print(f"{class_name}: {weights}")
    console.print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
