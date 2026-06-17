"""
training/train_student.py

Train TinyPhoBERT Student with Multi-Level Knowledge Distillation.

Usage:
    # Full distillation (A4)
    python training/train_student.py --config configs/distillation_config.yaml

    # No distillation (A1 baseline)
    python training/train_student.py --config configs/distillation_config.yaml --no_kd

    # Custom weights
    python training/train_student.py --alpha 0.7 --beta 0.2 --gamma 0.1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from models.student import TinyPhoBERT, build_student_from_config
from models.distillation_balance import (
    MultiLevelDistillationLoss,
    DistillationTrainer,
    build_distillation_loss_from_config,
)
from utils.data_utils import load_vihsd_from_csv, build_datasets
from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device
from utils.logging_utils import ExperimentLogger
from utils.data_utils import get_class_weights

console = Console()


def train_epoch(
    distill_trainer: DistillationTrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_clip: float = 1.0,
    fp16: bool = False,
    scaler=None,
) -> dict:
    distill_trainer.student.train()
    total_losses = {
        "loss": 0.0, "loss_ce": 0.0,
        "loss_kd": 0.0, "loss_hidden": 0.0, "loss_att": 0.0,
    }
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        if fp16 and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                losses = distill_trainer.distill_step(batch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distill_trainer.student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = distill_trainer.distill_step(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(distill_trainer.student.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()

        # Get predictions from student
        with torch.no_grad():
            labels = batch["labels"].to(distill_trainer.device)
            input_ids = batch["input_ids"].to(distill_trainer.device)
            attention_mask = batch["attention_mask"].to(distill_trainer.device)
            out = distill_trainer.student(input_ids, attention_mask)
            preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{losses['loss'].item():.4f}"})

    n = len(dataloader)
    metrics = {k: v / n for k, v in total_losses.items()}
    task_metrics = compute_metrics(all_labels, all_preds)
    metrics.update(task_metrics)
    return metrics


@torch.no_grad()
def evaluate(
    student: TinyPhoBERT,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "Val",
) -> tuple:
    student.eval()
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"  {split_name}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = student(input_ids, attention_mask)
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds


def train(config: dict, run_name: Optional[str] = None) -> dict:
    """
    Main distillation training function.

    Args:
        config: Full distillation config dictionary.
        run_name: Optional override for the experiment run name.

    Returns:
        Test metrics dictionary.
    """
    training_cfg = config["training"]
    data_cfg = config["data"]
    log_cfg = config.get("logging", {})
    distill_cfg = config["distillation"]

    set_seed(data_cfg["seed"])
    device = get_device()

    actual_run_name = run_name or log_cfg.get("run_name", "distillation")
    output_dir = os.path.join(training_cfg["output_dir"], actual_run_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = ExperimentLogger(
        project_name=log_cfg.get("project_name", "TinyPhoBERT"),
        run_name=actual_run_name,
        log_dir=os.path.join(log_cfg.get("log_dir", "logs"), actual_run_name),
        use_wandb=log_cfg.get("use_wandb", False),
        use_tensorboard=log_cfg.get("use_tensorboard", True),
        config=config,
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    console.print("[bold cyan]Loading tokenizer...[/bold cyan]")
    tokenizer = get_teacher_tokenizer("vinai/phobert-base")

    # ── Data ───────────────────────────────────────────────────────────────────
    console.print("[bold cyan]Loading ViHSD dataset...[/bold cyan]")
    train_df, val_df, test_df = load_vihsd_from_csv(
        data_cfg["train_file"], data_cfg["val_file"], data_cfg["test_file"],
        text_col=data_cfg["text_col"], label_col=data_cfg["label_col"],
    )
    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df, tokenizer,
        text_col=data_cfg["text_col"],
        label_col=data_cfg["label_col"],
        max_length=training_cfg["max_seq_length"],
    )

    train_labels = train_df[data_cfg["label_col"]].astype(int).tolist()
    class_weights = get_class_weights(train_labels, num_classes=3).tolist()
    console.print(f"Class weights: {class_weights}")

    nw = training_cfg.get("dataloader_num_workers", 4)
    bs = training_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False, num_workers=nw)

    # ── Teacher ────────────────────────────────────────────────────────────────
    teacher_path = config["teacher"]["model_path"]
    console.print(f"[bold cyan]Loading Teacher from: {teacher_path}[/bold cyan]")
    if os.path.isfile(os.path.join(teacher_path, "../best_model.pt")) or \
       os.path.isfile(teacher_path + ".pt") or \
       os.path.isfile(os.path.join(teacher_path, "best_model.pt")):
        # Load from our custom checkpoint
        ckpt_file = (
            os.path.join(teacher_path, "best_model.pt")
            if os.path.isfile(os.path.join(teacher_path, "best_model.pt"))
            else teacher_path + ".pt"
        )
        teacher = PhoBERTTeacher.from_pretrained_checkpoint(ckpt_file)
    elif os.path.isdir(teacher_path):
        # HuggingFace format
        from transformers import AutoModelForSequenceClassification
        teacher = PhoBERTTeacher(model_name=teacher_path)
    else:
        console.print(
            f"[yellow]Warning: Teacher checkpoint not found at '{teacher_path}'. "
            "Loading fresh PhoBERT-base (results will be suboptimal). "
            "Please run training/train_teacher.py first.[/yellow]"
        )
        teacher = PhoBERTTeacher("vinai/phobert-base")

    teacher.freeze()
    teacher_params = teacher.count_parameters()
    console.print(f"  Teacher params: [bold]{teacher_params:,}[/bold]")

    # ── Student ────────────────────────────────────────────────────────────────
    console.print("[bold cyan]Building TinyPhoBERT Student...[/bold cyan]")
    student = build_student_from_config(config.get("student", {}))
    student_params = student.count_parameters()
    student_mb = student.model_size_mb()
    student.print_summary()

    # ── Distillation Loss ──────────────────────────────────────────────────────
    distill_loss = MultiLevelDistillationLoss(
        alpha=distill_cfg["alpha"],
        beta=distill_cfg["beta"],
        gamma=distill_cfg["gamma"],
        temperature=distill_cfg["temperature"],
        class_weights=class_weights,
        use_logit_kd=distill_cfg["use_logit_kd"],
        use_hidden_kd=distill_cfg["use_hidden_kd"],
        use_attention_kd=distill_cfg["use_attention_kd"],
    )
    console.print(
        f"  Distillation: KD={distill_cfg['use_logit_kd']} | "
        f"Hidden={distill_cfg['use_hidden_kd']} | "
        f"Att={distill_cfg['use_attention_kd']}\n"
        f"  Weights: α={distill_cfg['alpha']} β={distill_cfg['beta']} γ={distill_cfg['gamma']} "
        f"T={distill_cfg['temperature']}"
    )

    distill_trainer = DistillationTrainer(teacher, student, distill_loss, device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    num_steps = len(train_loader) * training_cfg["num_epochs"]
    num_warmup = int(num_steps * training_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_steps)

    fp16 = training_cfg.get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if fp16 else None

    # ── Training Loop ──────────────────────────────────────────────────────────
    best_f1 = 0.0
    best_epoch = 0
    global_step = 0

    console.print(
        f"\n[bold cyan]Starting Distillation for {training_cfg['num_epochs']} epochs...[/bold cyan]\n"
    )

    for epoch in range(1, training_cfg["num_epochs"] + 1):
        console.print(f"[bold]Epoch {epoch}/{training_cfg['num_epochs']}[/bold]")

        train_metrics = train_epoch(
            distill_trainer, train_loader, optimizer, scheduler,
            grad_clip=training_cfg["max_grad_norm"], fp16=fp16, scaler=scaler,
        )
        global_step += len(train_loader)

        val_metrics, val_labels, val_preds = evaluate(student, val_loader, device, "Val")

        logger.log({
            "train/loss": train_metrics["loss"],
            "train/loss_ce": train_metrics["loss_ce"],
            "train/loss_kd": train_metrics["loss_kd"],
            "train/loss_hidden": train_metrics["loss_hidden"],
            "train/loss_att": train_metrics["loss_att"],
            "train/f1_macro": train_metrics["macro_f1"],
            "val/f1_macro": val_metrics["macro_f1"],
            "val/accuracy": val_metrics["accuracy"],
        }, step=global_step)

        console.print(
            f"  Train: loss={train_metrics['loss']:.4f} "
            f"(ce={train_metrics['loss_ce']:.4f} "
            f"kd={train_metrics['loss_kd']:.4f} "
            f"hid={train_metrics['loss_hidden']:.4f} "
            f"att={train_metrics['loss_att']:.4f})\n"
            f"  Val  : f1={val_metrics['macro_f1']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "val_f1": best_f1,
                    "run_name": actual_run_name,
                    "config": config,
                },
                os.path.join(output_dir, "best_model.pt"),
            )
            console.print(f"  [bold green]✓ Best model saved (F1={best_f1:.4f})[/bold green]")

    console.print(
        f"\n[bold green]Distillation complete! Best Val F1={best_f1:.4f} at epoch {best_epoch}[/bold green]"
    )

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    console.print("\n[bold cyan]Final Test Evaluation...[/bold cyan]")
    ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    student.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_labels, test_preds = evaluate(student, test_loader, device, "Test")
    print_classification_report(test_labels, test_preds)

    table = Table(title=f"TinyPhoBERT [{actual_run_name}] Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    test_metrics["run_name"] = actual_run_name
    test_metrics["params"] = student_params
    test_metrics["size_mb"] = round(student_mb, 2)
    with open(os.path.join(results_dir, f"{actual_run_name}_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.finish()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train TinyPhoBERT Student with Multi-Level KD.")
    parser.add_argument("--config", type=str, default="configs/distillation_config.yaml")
    parser.add_argument("--run_name", type=str, default=None, help="Experiment run name.")
    parser.add_argument("--alpha", type=float, default=None, help="Override KD loss weight.")
    parser.add_argument("--beta", type=float, default=None, help="Override Hidden KD weight.")
    parser.add_argument("--gamma", type=float, default=None, help="Override Attention KD weight.")
    parser.add_argument("--no_kd", action="store_true", help="Disable all distillation (A1 baseline).")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.no_kd:
        config["distillation"]["use_logit_kd"] = False
        config["distillation"]["use_hidden_kd"] = False
        config["distillation"]["use_attention_kd"] = False
        config["distillation"]["alpha"] = 0.0
        config["distillation"]["beta"] = 0.0
        config["distillation"]["gamma"] = 0.0

    if args.alpha is not None:
        config["distillation"]["alpha"] = args.alpha
    if args.beta is not None:
        config["distillation"]["beta"] = args.beta
    if args.gamma is not None:
        config["distillation"]["gamma"] = args.gamma
    if args.fp16:
        config["training"]["fp16"] = True

    train(config, run_name=args.run_name)


if __name__ == "__main__":
    main()
