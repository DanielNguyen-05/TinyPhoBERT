"""
evaluation/evaluate.py

Comprehensive evaluation of trained models on ViHSD test set.

Outputs:
    - Accuracy, Macro P/R/F1, Weighted F1
    - Per-class F1 (CLEAN, OFFENSIVE, HATE)
    - Confusion matrix
    - Classification report

Usage:
    # Evaluate TinyPhoBERT student
    python evaluation/evaluate.py --model_type student \
        --model_path checkpoints/distillation/distillation-full/best_model.pt

    # Evaluate Teacher
    python evaluation/evaluate.py --model_type teacher \
        --model_path checkpoints/teacher/best_model.pt

    # Compare all saved models
    python evaluation/evaluate.py --compare_all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from models.student import TinyPhoBERT, build_student_from_config
from utils.data_utils import load_vihsd_from_csv, build_datasets, HateSpeechDataset
from utils.metrics import (
    compute_metrics,
    print_classification_report,
    get_confusion_matrix,
    LABEL_NAMES,
)
from utils.seed import set_seed, get_device

console = Console()


@torch.no_grad()
def run_inference(model, dataloader, device, model_type="student"):
    """Run model inference and return predictions and true labels."""
    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        if model_type == "teacher":
            outputs = model(input_ids, attention_mask)
        else:
            outputs = model(input_ids, attention_mask)

        logits = outputs["logits"].cpu()
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_logits.append(logits.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    return all_labels, all_preds, all_logits


def plot_confusion_matrix(
    y_true, y_pred, label_names=LABEL_NAMES, title="Confusion Matrix", save_path=None
):
    """Plot and save confusion matrix."""
    cm = get_confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=label_names, yticklabels=label_names)
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=label_names, yticklabels=label_names)
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"  Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate_model(
    model,
    test_df,
    tokenizer,
    device,
    model_type="student",
    max_length=128,
    batch_size=64,
    model_name="Model",
    save_plots=True,
    plot_dir="results/plots",
) -> dict:
    """Evaluate a model on the test set."""
    test_ds = HateSpeechDataset(
        test_df["free_text"].astype(str).tolist(),
        test_df["label_id"].astype(int).tolist(),
        tokenizer,
        max_length,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    labels, preds, logits = run_inference(model, test_loader, device, model_type)
    metrics = compute_metrics(labels, preds)

    console.print(f"\n[bold]── {model_name} Evaluation Results ──[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)
    print_classification_report(labels, preds)

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        plot_confusion_matrix(
            labels, preds,
            title=model_name,
            save_path=os.path.join(plot_dir, f"{model_name.lower().replace(' ', '_')}_cm.png"),
        )

    return metrics


def compare_all_results(results_dir="results"):
    """Load and compare all saved result JSON files."""
    result_files = list(Path(results_dir).glob("*.json"))
    if not result_files:
        console.print(f"[yellow]No result files found in {results_dir}/[/yellow]")
        return

    all_results = {}
    for f in sorted(result_files):
        with open(f) as fp:
            data = json.load(fp)
        name = data.get("run_name", f.stem)
        all_results[name] = data

    table = Table(title="All Models — Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Macro-F1", style="green")
    table.add_column("Accuracy", style="blue")
    table.add_column("Macro-P", style="yellow")
    table.add_column("Macro-R", style="yellow")
    table.add_column("Params", style="magenta")
    table.add_column("Size (MB)", style="magenta")

    for name, m in sorted(all_results.items(), key=lambda x: x[1].get("macro_f1", 0), reverse=True):
        table.add_row(
            name,
            f"{m.get('macro_f1', 0):.4f}",
            f"{m.get('accuracy', 0):.4f}",
            f"{m.get('macro_precision', 0):.4f}",
            f"{m.get('macro_recall', 0):.4f}",
            str(m.get("params", "N/A")),
            str(m.get("size_mb", "N/A")),
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on ViHSD test set.")
    parser.add_argument("--model_type", choices=["teacher", "student"], default="student")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="configs/student_config.yaml")
    parser.add_argument(
        "--test_file", type=str, default="data/augmented/test.csv",
        help="Test set ĐÃ preprocess (cùng pipeline với lúc train). "
             "KHÔNG dùng data/processed/test.csv (thô) vì model được "
             "train trên text đã preprocess — đánh giá trên text thô "
             "sẽ cho kết quả sai lệch do train/inference mismatch.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--compare_all", action="store_true",
                        help="Compare all saved results.")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    if args.compare_all:
        compare_all_results()
        return

    if args.model_path is None:
        console.print("[red]Error: --model_path is required.[/red]")
        sys.exit(1)

    # Load data
    test_df = pd.read_csv(args.test_file)
    console.print(f"[cyan]Test set: {len(test_df)} samples[/cyan]")

    tokenizer = get_teacher_tokenizer("vinai/phobert-base")

    # Load model
    if args.model_type == "teacher":
        model = PhoBERTTeacher()
        ckpt = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)
        model_name = "PhoBERT Teacher"
    else:
        import yaml
        with open(args.config_path) as f:
            cfg = yaml.safe_load(f)
        model = build_student_from_config(cfg)
        ckpt = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)
        model_name = ckpt.get("run_name", "TinyPhoBERT")

    metrics = evaluate_model(
        model, test_df, tokenizer, device,
        model_type=args.model_type,
        max_length=args.max_length,
        batch_size=args.batch_size,
        model_name=model_name,
        save_plots=not args.no_plots,
    )

    # Save
    os.makedirs("results", exist_ok=True)
    save_path = f"results/eval_{model_name.lower().replace(' ', '_')}.json"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()