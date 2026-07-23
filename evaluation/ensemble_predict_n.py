"""
evaluation/ensemble_predict_n.py

Kết hợp dự đoán từ N model độc lập (mở rộng từ ensemble_predict.py chỉ
hỗ trợ 2 model) — dùng grid search trên simplex trọng số để tìm tổ hợp
tối ưu trên VAL set, áp dụng lên TEST set 1 lần duy nhất.

Usage:
    python evaluation/ensemble_predict_n.py \\
        --model_dirs checkpoints/phobert_v2_fgm checkpoints/visobert_noaug checkpoints/qwen_classifier \\
        --model_names PhoBERT_v2 ViSoBERT Qwen
"""

import argparse
import itertools
import json
import os

import numpy as np
from rich.console import Console
from rich.table import Table

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

console = Console()

LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_nll = 1.0, float("inf")
    for temperature in np.arange(0.5, 3.01, 0.05):
        calibrated = temperature_scale_probs(probs, float(temperature))
        nll = -np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_t, best_nll = float(temperature), float(nll)
    return best_t


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(LABEL_NAMES):
        if i < len(per_class_f1):
            metrics[f"f1_{name.lower()}"] = per_class_f1[i]
    return metrics


def generate_simplex_weights(n_models: int, step: float = 0.1):
    """
    Sinh mọi tổ hợp trọng số (w1, w2, ..., wn) sao cho sum(w_i)=1, mỗi
    w_i là bội số của `step`. Với n=3, step=0.1 → 66 tổ hợp — đủ nhanh.
    """
    n_steps = int(round(1.0 / step))
    results = []
    for combo in itertools.product(range(n_steps + 1), repeat=n_models - 1):
        if sum(combo) > n_steps:
            continue
        last = n_steps - sum(combo)
        weights = tuple(c * step for c in combo) + (last * step,)
        results.append(weights)
    return results


def load_required(model_dir: str, label: str):
    paths = {
        "val_probs": os.path.join(model_dir, "val_probs.npy"),
        "val_labels": os.path.join(model_dir, "val_labels.npy"),
        "test_probs": os.path.join(model_dir, "test_probs.npy"),
        "test_labels": os.path.join(model_dir, "test_labels.npy"),
    }
    missing = [name for name, p in paths.items() if not os.path.isfile(p)]
    if missing:
        console.print(f"[bold red]✗ [{label}] Thiếu file trong '{model_dir}': {missing}[/bold red]")
        for name, p in paths.items():
            exists = "✓" if os.path.isfile(p) else "✗ KHÔNG TỒN TẠI"
            console.print(f"    {name}: {os.path.abspath(p)} [{exists}]")
        return None
    return {name: np.load(p) for name, p in paths.items()}


def main():
    parser = argparse.ArgumentParser(description="Ensemble N models đã train độc lập.")
    parser.add_argument("--model_dirs", type=str, nargs="+", required=True,
                         help="Danh sách thư mục chứa val/test probs.npy, cách nhau bằng dấu cách")
    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                         help="Tên hiển thị cho mỗi model (mặc định dùng tên thư mục)")
    parser.add_argument("--metric", type=str, default="macro_f1")
    parser.add_argument("--weight_step", type=float, default=0.1,
                         help="Bước nhảy trọng số. Với >=3 model nên dùng 0.1 để tránh quá chậm.")
    parser.add_argument("--no_temperature_calibration", action="store_true")
    args = parser.parse_args()

    n_models = len(args.model_dirs)
    if args.model_names is None:
        names = [os.path.basename(d.rstrip("/")) for d in args.model_dirs]
    else:
        names = args.model_names
        assert len(names) == n_models, "Số lượng --model_names phải khớp --model_dirs"

    console.print(f"[bold cyan]Loading probabilities cho {n_models} models...[/bold cyan]")
    all_data = []
    for d, name in zip(args.model_dirs, names):
        data = load_required(d, name)
        if data is None:
            return
        all_data.append(data)

    # ── Sanity check: labels khớp giữa mọi model ─────────────────────────────
    ref_val_labels = all_data[0]["val_labels"]
    ref_test_labels = all_data[0]["test_labels"]
    for data, name in zip(all_data[1:], names[1:]):
        if not np.array_equal(ref_val_labels, data["val_labels"]):
            console.print(f"[bold red]✗ Val labels của '{name}' KHÔNG khớp model đầu tiên![/bold red]")
            return
        if not np.array_equal(ref_test_labels, data["test_labels"]):
            console.print(f"[bold red]✗ Test labels của '{name}' KHÔNG khớp model đầu tiên![/bold red]")
            return

    console.print(f"  Val: {len(ref_val_labels)} samples | Test: {len(ref_test_labels)} samples | labels khớp ✓")

    temperatures = [1.0] * n_models
    if not args.no_temperature_calibration:
        temperatures = [
            fit_temperature(data["val_probs"], ref_val_labels) for data in all_data
        ]
        for data, temperature in zip(all_data, temperatures):
            data["val_probs"] = temperature_scale_probs(data["val_probs"], temperature)
            data["test_probs"] = temperature_scale_probs(data["test_probs"], temperature)
        console.print(
            "  Validation temperatures: "
            + ", ".join(f"{name}={temp:.2f}" for name, temp in zip(names, temperatures))
        )

    # ── Baseline: từng model riêng lẻ ─────────────────────────────────────────
    individual_metrics = []
    for data, name in zip(all_data, names):
        preds = data["test_probs"].argmax(axis=-1)
        m = compute_metrics(ref_test_labels, preds)
        individual_metrics.append(m)

    # ── Grid search trọng số trên VAL set ──────────────────────────────────────
    console.print(f"\n[bold cyan]Grid search trọng số ({n_models} models, step={args.weight_step})...[/bold cyan]")
    weight_combos = generate_simplex_weights(n_models, step=args.weight_step)
    console.print(f"  Tổng số tổ hợp cần thử: {len(weight_combos)}")

    val_probs_stack = [d["val_probs"] for d in all_data]
    best_weights, best_score = None, -1.0
    for weights in weight_combos:
        combined = sum(w * p for w, p in zip(weights, val_probs_stack))
        preds = combined.argmax(axis=-1)
        score = compute_metrics(ref_val_labels, preds)[args.metric]
        if score > best_score:
            best_score = score
            best_weights = weights

    console.print(f"  Best weights: {dict(zip(names, [f'{w:.2f}' for w in best_weights]))}")
    console.print(f"  Val {args.metric} = {best_score:.4f}")

    # ── Áp dụng lên TEST set (chỉ 1 lần) ───────────────────────────────────────
    test_probs_stack = [d["test_probs"] for d in all_data]
    ensemble_probs = sum(w * p for w, p in zip(best_weights, test_probs_stack))
    ensemble_preds = ensemble_probs.argmax(axis=-1)
    ensemble_metrics = compute_metrics(ref_test_labels, ensemble_preds)

    table = Table(title=f"So sánh {n_models} Models vs Ensemble (Test set)")
    table.add_column("Metric", style="cyan")
    for name in names:
        table.add_column(name, style="yellow")
    table.add_column("Ensemble", style="bold green")

    for key in ["accuracy", "macro_f1", "f1_clean", "f1_offensive", "f1_hate"]:
        row = [key] + [f"{m.get(key, 0):.4f}" for m in individual_metrics] + [f"{ensemble_metrics.get(key, 0):.4f}"]
        table.add_row(*row)
    console.print("\n")
    console.print(table)

    best_individual = max(m["macro_f1"] for m in individual_metrics)
    improvement = ensemble_metrics["macro_f1"] - best_individual
    if improvement > 0.005:
        console.print(f"\n[bold green]✓ Ensemble CẢI THIỆN {improvement:+.4f} Macro-F1 so với model tốt nhất riêng lẻ.[/bold green]")
    elif improvement > -0.005:
        console.print(f"\n[bold yellow]≈ Ensemble tương đương model tốt nhất riêng lẻ ({improvement:+.4f}).[/bold yellow]")
    else:
        console.print(f"\n[bold red]✗ Ensemble KÉM HƠN model tốt nhất riêng lẻ ({improvement:+.4f}).[/bold red]")

    os.makedirs("results", exist_ok=True)
    with open("results/ensemble_n_results.json", "w") as f:
        json.dump({
            "model_names": names,
            "individual_metrics": individual_metrics,
            "ensemble_metrics": ensemble_metrics,
            "best_weights": dict(zip(names, best_weights)),
            "val_score_at_best_weights": best_score,
            "temperatures": dict(zip(names, temperatures)),
        }, f, indent=2)
    console.print("\nResults saved to results/ensemble_n_results.json")


if __name__ == "__main__":
    main()
