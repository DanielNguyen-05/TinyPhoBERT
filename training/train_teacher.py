"""
training/train_teacher.py

Fine-tune PhoBERT-base (Teacher) on the ViHSD dataset.

Usage:
    python training/train_teacher.py
    python training/train_teacher.py --config configs/teacher_config.yaml
    python training/train_teacher.py --config configs/teacher_config.yaml --fp16
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from utils.data_utils import load_vihsd_from_csv, build_datasets
from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device
from utils.logging_utils import ExperimentLogger

console = Console()


def train_epoch(
    model: PhoBERTTeacher,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
    fp16: bool = False,
    scaler=None,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if fp16 and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def evaluate(
    model: PhoBERTTeacher,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "Val",
) -> dict:
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


def train(config: dict) -> None:
    # Setup
    set_seed(config["data"]["seed"])
    device = get_device()
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    log_cfg = config.get("logging", {})
    logger = ExperimentLogger(
        project_name=log_cfg.get("project_name", "TinyPhoBERT"),
        run_name=log_cfg.get("run_name", "teacher"),
        log_dir=log_cfg.get("log_dir", "logs/teacher"),
        use_wandb=log_cfg.get("use_wandb", False),
        use_tensorboard=log_cfg.get("use_tensorboard", True),
        config=config,
    )

    # Tokenizer & Data
    console.print("[bold cyan]Loading tokenizer and data...[/bold cyan]")
    tokenizer = get_teacher_tokenizer(config["model"]["name"])

    train_df, val_df, test_df = load_vihsd_from_csv(
        train_path=config["data"]["train_file"],
        val_path=config["data"]["val_file"],
        test_path=config["data"]["test_file"],
        text_col=config["data"]["text_col"],
        label_col=config["data"]["label_col"],
    )

    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df, tokenizer,
        text_col=config["data"]["text_col"],
        label_col=config["data"]["label_col"],
        max_length=config["training"]["max_seq_length"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("dataloader_num_workers", 4),
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=config["training"].get("dataloader_num_workers", 4),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=config["training"].get("dataloader_num_workers", 4),
    )

    # Model
    console.print("[bold cyan]Building Teacher model (PhoBERT-base)...[/bold cyan]")
    model = PhoBERTTeacher(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    ).to(device)
    n_params = model.count_parameters()
    console.print(f"  Teacher params: [bold green]{n_params:,}[/bold green]")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    num_warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    fp16 = config["training"].get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    if fp16:
        console.print("  [yellow]FP16 mixed precision enabled.[/yellow]")

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    global_step = 0

    console.print(f"\n[bold cyan]Starting Teacher training for {config['training']['num_epochs']} epochs...[/bold cyan]\n")

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        console.print(f"[bold]Epoch {epoch}/{config['training']['num_epochs']}[/bold]")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=config["training"]["max_grad_norm"],
            fp16=fp16, scaler=scaler,
        )
        global_step += len(train_loader)

        val_metrics, val_labels, val_preds = evaluate(model, val_loader, device, "Val")

        # Log
        log_dict = {
            "train/loss": train_metrics["loss"],
            "train/f1_macro": train_metrics["macro_f1"],
            "val/loss": val_metrics["loss"],
            "val/f1_macro": val_metrics["macro_f1"],
            "val/accuracy": val_metrics["accuracy"],
        }
        logger.log(log_dict, step=global_step)

        console.print(
            f"  Train: loss={train_metrics['loss']:.4f} | f1={train_metrics['macro_f1']:.4f}\n"
            f"  Val  : loss={val_metrics['loss']:.4f}   | f1={val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            ckpt_dir = config["training"]["output_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            # Save full model state
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_f1,
                    "config": config,
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
            # Also save HuggingFace-compatible
            model.backbone.save_pretrained(os.path.join(ckpt_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(ckpt_dir, "best_model"))
            console.print(f"  [bold green]✓ Best model saved (F1={best_f1:.4f})[/bold green]")

    console.print(f"\n[bold green]Training complete! Best Val F1={best_f1:.4f} at epoch {best_epoch}[/bold green]")

    # Final test evaluation
    console.print("\n[bold cyan]Final Test Evaluation...[/bold cyan]")
    # Load best model
    ckpt = torch.load(os.path.join(config["training"]["output_dir"], "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_labels, test_preds = evaluate(model, test_loader, device, "Test")
    console.print("\n[bold]Test Results:[/bold]")
    print_classification_report(test_labels, test_preds)

    # Save results table
    table = Table(title="Teacher (PhoBERT-base) Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    # Save results to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    import json
    with open(os.path.join(results_dir, "teacher_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    console.print(f"Results saved to {results_dir}/teacher_results.json")

    logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT Teacher on ViHSD.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/teacher_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.fp16:
        config["training"]["fp16"] = True

    train(config)


if __name__ == "__main__":
    main()
