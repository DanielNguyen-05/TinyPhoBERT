"""
evaluation/ensemble_predict.py

Kết hợp dự đoán từ 2 model ĐỘC LẬP (PhoBERT-large Teacher +
Qwen2.5 QLoRA classifier) bằng weighted average của softmax probabilities.

KHÁC BIỆT so với LLM Fusion (đã thử, không hiệu quả):
    - LLM Fusion: kết hợp EMBEDDING trong lúc training — 2 model học chung.
    - Ensemble (script này): kết hợp PREDICTION sau khi mỗi model đã train
      XONG, ĐỘC LẬP — không training chung, không rủi ro can thiệp lẫn nhau.

Quy trình (tránh data leakage):
    1. Tìm trọng số w tối ưu bằng grid search trên VAL set
    2. Áp dụng w cố định đó lên TEST set — chỉ đánh giá 1 lần duy nhất

Yêu cầu trước khi chạy:
    python evaluation/save_teacher_probs.py --checkpoint ... --config ...
    python training/train_qwen_classifier.py --config configs/qwen_config.yaml

Usage:
    python evaluation/ensemble_predict.py \\
        --model_a_dir checkpoints/phobert_base \\
        --model_b_dir checkpoints/visobert
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


def find_best_weight(teacher_probs, qwen_probs, labels, metric="macro_f1", step=0.05):
    weights = np.arange(0.0, 1.0 + step, step)
    results = []
    for w in weights:
        combined = w * teacher_probs + (1 - w) * qwen_probs
        preds = combined.argmax(axis=-1)
        m = compute_metrics(labels, preds)
        results.append((w, m[metric]))
    best_w, best_score = max(results, key=lambda x: x[1])
    return best_w, best_score, results


def main():
    parser = argparse.ArgumentParser(description="Ensemble Teacher + Qwen predictions.")
    parser.add_argument("--model_a_dir", type=str, default="checkpoints/phobert_base")
    parser.add_argument("--model_b_dir", type=str, default="checkpoints/visobert")
    parser.add_argument("--metric", type=str, default="macro_f1")
    parser.add_argument("--weight_step", type=float, default=0.05)
    args = parser.parse_args()

    console.print("[bold cyan]Loading saved probabilities...[/bold cyan]")

    required_files = {
        "model_a_val_probs": os.path.join(args.model_a_dir, "val_probs.npy"),
        "model_a_val_labels": os.path.join(args.model_a_dir, "val_labels.npy"),
        "model_a_test_probs": os.path.join(args.model_a_dir, "test_probs.npy"),
        "model_a_test_labels": os.path.join(args.model_a_dir, "test_labels.npy"),
        "model_b_val_probs": os.path.join(args.model_b_dir, "val_probs.npy"),
        "model_b_val_labels": os.path.join(args.model_b_dir, "val_labels.npy"),
        "model_b_test_probs": os.path.join(args.model_b_dir, "test_probs.npy"),
        "model_b_test_labels": os.path.join(args.model_b_dir, "test_labels.npy"),
    }

    missing = [k for k, v in required_files.items() if not os.path.isfile(v)]
    if missing:
        console.print(f"[bold red]✗ Thiếu file: {missing}[/bold red]")
        console.print(
            "[yellow]Chạy trước:\n"
            "  python evaluation/save_teacher_probs.py --checkpoint ... --config ...\n"
            "  python training/train_qwen_classifier.py --config configs/qwen_config.yaml[/yellow]"
        )
        return

    model_a_val_probs = np.load(required_files["model_a_val_probs"])
    model_a_val_labels = np.load(required_files["model_a_val_labels"])
    model_a_test_probs = np.load(required_files["model_a_test_probs"])
    model_a_test_labels = np.load(required_files["model_a_test_labels"])

    model_b_val_probs = np.load(required_files["model_b_val_probs"])
    model_b_val_labels = np.load(required_files["model_b_val_labels"])
    model_b_test_probs = np.load(required_files["model_b_test_probs"])
    model_b_test_labels = np.load(required_files["model_b_test_labels"])

    if not np.array_equal(model_a_val_labels, model_b_val_labels):
        console.print(
            "[bold red]✗ Val labels giữa Teacher và Qwen KHÔNG khớp!\n"
            "  Có thể do 2 model dùng data/augmented khác version, hoặc\n"
            "  thứ tự samples khác nhau. Ensemble không hợp lệ.[/bold red]"
        )
        return
    if not np.array_equal(model_a_test_labels, model_b_test_labels):
        console.print("[bold red]✗ Test labels giữa Teacher và Qwen KHÔNG khớp![/bold red]")
        return

    console.print(f"  Val: {len(model_a_val_labels)} samples | Test: {len(model_a_test_labels)} samples | labels khớp ✓")

    model_a_test_preds = model_a_test_probs.argmax(axis=-1)
    model_b_test_preds = model_b_test_probs.argmax(axis=-1)

    model_a_metrics = compute_metrics(model_a_test_labels, model_a_test_preds)
    model_b_metrics = compute_metrics(model_a_test_labels, model_b_test_preds)

    console.print(f"\n[bold cyan]Grid search trọng số w trên VAL set (metric={args.metric})...[/bold cyan]")
    best_w, best_val_score, all_results = find_best_weight(
        model_a_val_probs, model_b_val_probs, model_a_val_labels, metric=args.metric, step=args.weight_step,
    )
    console.print(f"  Best w={best_w:.2f} (Model A weight) | Val {args.metric}={best_val_score:.4f}")

    console.print("\n  [dim]w=1.0 (Teacher only) | w=0.0 (Qwen only) | w=best[/dim]")
    for w, score in all_results:
        if abs(w - 1.0) < 1e-6 or abs(w - 0.0) < 1e-6 or abs(w - best_w) < 1e-6:
            marker = " ← best" if abs(w - best_w) < 1e-6 else ""
            console.print(f"    w={w:.2f}: {args.metric}={score:.4f}{marker}")

    ensemble_test_probs = best_w * model_a_test_probs + (1 - best_w) * model_b_test_probs
    ensemble_test_preds = ensemble_test_probs.argmax(axis=-1)
    ensemble_metrics = compute_metrics(model_a_test_labels, ensemble_test_preds)

    table = Table(title="So sánh: Teacher vs Qwen vs Ensemble (Test set)")
    table.add_column("Metric", style="cyan")
    table.add_column("Model A", style="yellow")
    table.add_column("Model B", style="yellow")
    table.add_column("Ensemble", style="bold green")

    for key in ["accuracy", "macro_f1", "f1_clean", "f1_offensive", "f1_hate"]:
        table.add_row(key, f"{model_a_metrics.get(key, 0):.4f}", f"{model_b_metrics.get(key, 0):.4f}", f"{ensemble_metrics.get(key, 0):.4f}")
    console.print("\n")
    console.print(table)

    improvement = ensemble_metrics["macro_f1"] - max(model_a_metrics["macro_f1"], model_b_metrics["macro_f1"])
    if improvement > 0.005:
        console.print(f"\n[bold green]✓ Ensemble CẢI THIỆN {improvement:+.4f} Macro-F1 so với model tốt nhất riêng lẻ.[/bold green]")
    elif improvement > -0.005:
        console.print(f"\n[bold yellow]≈ Ensemble tương đương model tốt nhất riêng lẻ ({improvement:+.4f}).[/bold yellow]")
    else:
        console.print(f"\n[bold red]✗ Ensemble KÉM HƠN model tốt nhất riêng lẻ ({improvement:+.4f}).[/bold red]")

    os.makedirs("results", exist_ok=True)
    with open("results/ensemble_results.json", "w") as f:
        json.dump({
            "model_a_metrics": model_a_metrics,
            "model_b_metrics": model_b_metrics,
            "ensemble_metrics": ensemble_metrics,
            "best_weight_teacher": float(best_w),
            "best_weight_qwen": float(1 - best_w),
            "val_score_at_best_w": float(best_val_score),
        }, f, indent=2)
    console.print("\nResults saved to results/ensemble_results.json")


if __name__ == "__main__":
    main()