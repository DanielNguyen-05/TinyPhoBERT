"""
evaluation/ensemble_predict.py

Kết hợp dự đoán từ 2 model ĐỘC LẬP bằng weighted average của softmax
probabilities. Viết lại sạch (không qua sed rename) để tránh lỗi ẩn.

Quy trình (tránh data leakage):
    1. Tìm trọng số w tối ưu bằng grid search trên VAL set
    2. Áp dụng w cố định đó lên TEST set — chỉ đánh giá 1 lần duy nhất

Yêu cầu trước khi chạy: mỗi model_dir phải có sẵn 4 file:
    val_probs.npy, val_labels.npy, test_probs.npy, test_labels.npy
(tạo bằng evaluation/save_teacher_probs.py)

Usage:
    python evaluation/ensemble_predict.py \\
        --model_a_dir checkpoints/phobert_v2_fgm \\
        --model_b_dir checkpoints/visobert_noaug
"""

import argparse
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
    """Select probability temperature on validation NLL only."""
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


def find_best_weight(probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray,
                      metric: str = "macro_f1", step: float = 0.05):
    weights = np.arange(0.0, 1.0 + step, step)
    results = []
    for w in weights:
        combined = w * probs_a + (1 - w) * probs_b
        preds = combined.argmax(axis=-1)
        m = compute_metrics(labels, preds)
        results.append((w, m[metric]))
    best_w, best_score = max(results, key=lambda x: x[1])
    return best_w, best_score, results


def load_required(model_dir: str, label: str) -> dict:
    """Load 4 file .npy cho 1 model, báo lỗi rõ ràng nếu thiếu file nào."""
    paths = {
        "val_probs": os.path.join(model_dir, "val_probs.npy"),
        "val_labels": os.path.join(model_dir, "val_labels.npy"),
        "test_probs": os.path.join(model_dir, "test_probs.npy"),
        "test_labels": os.path.join(model_dir, "test_labels.npy"),
    }

    missing = [name for name, p in paths.items() if not os.path.isfile(p)]
    if missing:
        console.print(f"[bold red]✗ [{label}] Thiếu file trong '{model_dir}': {missing}[/bold red]")
        console.print(f"[yellow]  Kiểm tra: ls -la {model_dir}[/yellow]")
        console.print(f"[yellow]  Đường dẫn đầy đủ đã kiểm tra:[/yellow]")
        for name, p in paths.items():
            exists = "✓" if os.path.isfile(p) else "✗ KHÔNG TỒN TẠI"
            console.print(f"    {name}: {os.path.abspath(p)} [{exists}]")
        return None

    return {name: np.load(p) for name, p in paths.items()}


def main():
    parser = argparse.ArgumentParser(description="Ensemble 2 models đã train độc lập.")
    parser.add_argument("--model_a_dir", type=str, required=True,
                         help="Thư mục chứa val/test probs.npy của model A")
    parser.add_argument("--model_b_dir", type=str, required=True,
                         help="Thư mục chứa val/test probs.npy của model B")
    parser.add_argument("--metric", type=str, default="macro_f1")
    parser.add_argument("--weight_step", type=float, default=0.05)
    parser.add_argument("--no_temperature_calibration", action="store_true")
    args = parser.parse_args()

    console.print("[bold cyan]Loading saved probabilities...[/bold cyan]")
    console.print(f"  Model A dir: {os.path.abspath(args.model_a_dir)}")
    console.print(f"  Model B dir: {os.path.abspath(args.model_b_dir)}")

    data_a = load_required(args.model_a_dir, "Model A")
    data_b = load_required(args.model_b_dir, "Model B")

    if data_a is None or data_b is None:
        console.print(
            "\n[yellow]Chạy trước khi ensemble:\n"
            "  python evaluation/save_teacher_probs.py --checkpoint <ckpt> "
            "--config <config> --output_dir <thư mục tương ứng>[/yellow]"
        )
        return

    # ── Sanity check: đảm bảo 2 model đánh giá trên CÙNG samples ────────────
    if not np.array_equal(data_a["val_labels"], data_b["val_labels"]):
        console.print("[bold red]✗ Val labels giữa Model A và Model B KHÔNG khớp! Ensemble không hợp lệ.[/bold red]")
        return
    if not np.array_equal(data_a["test_labels"], data_b["test_labels"]):
        console.print("[bold red]✗ Test labels giữa Model A và Model B KHÔNG khớp![/bold red]")
        return

    temperatures = [1.0, 1.0]
    if not args.no_temperature_calibration:
        temperatures = [
            fit_temperature(data_a["val_probs"], data_a["val_labels"]),
            fit_temperature(data_b["val_probs"], data_b["val_labels"]),
        ]
        for data, temperature in zip([data_a, data_b], temperatures):
            data["val_probs"] = temperature_scale_probs(data["val_probs"], temperature)
            data["test_probs"] = temperature_scale_probs(data["test_probs"], temperature)
        console.print(f"  Validation temperatures: A={temperatures[0]:.2f}, B={temperatures[1]:.2f}")

    console.print(
        f"  Val: {len(data_a['val_labels'])} samples | "
        f"Test: {len(data_a['test_labels'])} samples | labels khớp ✓"
    )

    # ── Baseline: từng model riêng lẻ trên test ──────────────────────────────
    preds_a_test = data_a["test_probs"].argmax(axis=-1)
    preds_b_test = data_b["test_probs"].argmax(axis=-1)

    metrics_a = compute_metrics(data_a["test_labels"], preds_a_test)
    metrics_b = compute_metrics(data_b["test_labels"], preds_b_test)

    # ── Tìm trọng số tối ưu trên VAL set (KHÔNG dùng test) ──────────────────
    console.print(f"\n[bold cyan]Grid search trọng số w trên VAL set (metric={args.metric})...[/bold cyan]")
    best_w, best_val_score, all_results = find_best_weight(
        data_a["val_probs"], data_b["val_probs"], data_a["val_labels"],
        metric=args.metric, step=args.weight_step,
    )
    console.print(f"  Best w={best_w:.2f} (Model A weight) | Val {args.metric}={best_val_score:.4f}")

    console.print("\n  [dim]w=1.0 (Model A only) | w=0.0 (Model B only) | w=best[/dim]")
    for w, score in all_results:
        if abs(w - 1.0) < 1e-6 or abs(w - 0.0) < 1e-6 or abs(w - best_w) < 1e-6:
            marker = " ← best" if abs(w - best_w) < 1e-6 else ""
            console.print(f"    w={w:.2f}: {args.metric}={score:.4f}{marker}")

    # ── Áp dụng w tối ưu lên TEST set (chỉ 1 lần) ────────────────────────────
    ensemble_probs = best_w * data_a["test_probs"] + (1 - best_w) * data_b["test_probs"]
    ensemble_preds = ensemble_probs.argmax(axis=-1)
    ensemble_metrics = compute_metrics(data_a["test_labels"], ensemble_preds)

    table = Table(title="So sánh: Model A vs Model B vs Ensemble (Test set)")
    table.add_column("Metric", style="cyan")
    table.add_column("Model A", style="yellow")
    table.add_column("Model B", style="yellow")
    table.add_column("Ensemble", style="bold green")

    for key in ["accuracy", "macro_f1", "f1_clean", "f1_offensive", "f1_hate"]:
        table.add_row(
            key,
            f"{metrics_a.get(key, 0):.4f}",
            f"{metrics_b.get(key, 0):.4f}",
            f"{ensemble_metrics.get(key, 0):.4f}",
        )
    console.print("\n")
    console.print(table)

    improvement = ensemble_metrics["macro_f1"] - max(metrics_a["macro_f1"], metrics_b["macro_f1"])
    if improvement > 0.005:
        console.print(f"\n[bold green]✓ Ensemble CẢI THIỆN {improvement:+.4f} Macro-F1 so với model tốt nhất riêng lẻ.[/bold green]")
    elif improvement > -0.005:
        console.print(f"\n[bold yellow]≈ Ensemble tương đương model tốt nhất riêng lẻ ({improvement:+.4f}).[/bold yellow]")
    else:
        console.print(f"\n[bold red]✗ Ensemble KÉM HƠN model tốt nhất riêng lẻ ({improvement:+.4f}).[/bold red]")

    os.makedirs("results", exist_ok=True)
    with open("results/ensemble_results.json", "w") as f:
        json.dump({
            "model_a_metrics": metrics_a,
            "model_b_metrics": metrics_b,
            "ensemble_metrics": ensemble_metrics,
            "best_weight_model_a": float(best_w),
            "best_weight_model_b": float(1 - best_w),
            "val_score_at_best_w": float(best_val_score),
            "temperatures": temperatures,
        }, f, indent=2)
    console.print("\nResults saved to results/ensemble_results.json")


if __name__ == "__main__":
    main()
