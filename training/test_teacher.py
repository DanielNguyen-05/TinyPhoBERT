"""
Evaluate saved PhoBERT Teacher checkpoint on ViHSD test set.

Usage:
    python training/test_teacher.py
    python training/test_teacher.py --config configs/teacher_config.yaml
    python training/test_teacher.py --checkpoint checkpoints/teacher/best_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from utils.data_utils import load_vihsd_from_csv, build_datasets, get_class_weights
from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device

console = Console()


@torch.no_grad()
def evaluate(model, dataloader, device, split_name="Test"):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"  {split_name}", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        total_loss += loss.item()

        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhoBERT Teacher checkpoint on test set.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/teacher_config.yaml",
        help="Path to teacher config YAML file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to best_model.pt checkpoint. If not provided, use output_dir/best_model.pt from config.",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["data"]["seed"])
    device = get_device()

    console.print("[bold cyan]Loading tokenizer and test data...[/bold cyan]")

    tokenizer = get_teacher_tokenizer(config["model"]["name"])

    train_df, val_df, test_df = load_vihsd_from_csv(
        train_path=config["data"]["train_file"],
        val_path=config["data"]["val_file"],
        test_path=config["data"]["test_file"],
        text_col=config["data"]["text_col"],
        label_col=config["data"]["label_col"],
    )

    train_ds, val_ds, test_ds = build_datasets(
        train_df,
        val_df,
        test_df,
        tokenizer,
        text_col=config["data"]["text_col"],
        label_col=config["data"]["label_col"],
        max_length=config["training"]["max_seq_length"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=config["training"].get("dataloader_num_workers", 4),
    )

    # Need class weights because your teacher model was initialized with Focal Loss
    train_labels = train_df[config["data"]["label_col"]].astype(int).tolist()
    class_weights = get_class_weights(
        train_labels,
        num_classes=config["model"]["num_labels"],
    ).to(device)

    console.print("[bold cyan]Building Teacher model...[/bold cyan]")

    use_focal = config["training"].get("use_focal_loss", True)
    focal_gamma = config["training"].get("focal_gamma", 2.0)
    label_smoothing = config["training"].get("label_smoothing", 0.1)

    model = PhoBERTTeacher(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
        class_weights=class_weights if use_focal else None,
        use_focal_loss=use_focal,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        classification_head=config["model"].get("classification_head", "linear"),
        num_mixed_layers=config["model"].get("num_mixed_layers", 4),
        cnn_kernel_sizes=tuple(config["model"].get("cnn_kernel_sizes", [1, 3, 5])),
        cnn_channels=config["model"].get("cnn_channels", 128),
    ).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config["training"]["output_dir"],
            "best_model.pt",
        )

    console.print(f"[bold cyan]Loading checkpoint:[/bold cyan] {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(ckpt["model_state_dict"])

    console.print("[bold green]Checkpoint loaded successfully.[/bold green]")

    if "epoch" in ckpt:
        console.print(f"Best epoch: [yellow]{ckpt['epoch']}[/yellow]")
    if "val_f1" in ckpt:
        console.print(f"Best val F1: [yellow]{ckpt['val_f1']:.4f}[/yellow]")

    console.print("\n[bold cyan]Final Test Evaluation...[/bold cyan]")

    test_metrics, test_labels, test_preds = evaluate(
        model,
        test_loader,
        device,
        "Test",
    )

    console.print("\n[bold]Test Results:[/bold]")
    print_classification_report(test_labels, test_preds)

    table = Table(title=f"Teacher ({config['model']['name']}) Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")

    console.print(table)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, "teacher_test_results.json")

    with open(save_path, "w") as f:
        json.dump(
            {
                **test_metrics,
                "checkpoint": checkpoint_path,
                "model_name": config["model"]["name"],
                "best_epoch": ckpt.get("epoch", None),
                "best_val_f1": float(ckpt["val_f1"]) if "val_f1" in ckpt else None,
            },
            f,
            indent=2,
        )

    console.print(f"[bold green]Results saved to {save_path}[/bold green]")


if __name__ == "__main__":
    main()
