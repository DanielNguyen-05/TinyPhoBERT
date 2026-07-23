"""
training/train_moe_distill.py

Multi-Teacher Knowledge Distillation: chưng cất kiến thức từ MoE teacher
(5 expert + gating network, đã lưu ở checkpoints/gating_network/moe_teacher_probs/)
vào TinyPhoBERT student gọn nhẹ.

Usage:
    python training/train_moe_distill.py \\
        --moe_teacher_dir checkpoints/gating_network/moe_teacher_probs \\
        --config configs/teacher_config.yaml \\
        --output_dir checkpoints/tinyphobert_moe_distilled
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student import TinyPhoBERT

console = Console()
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def compute_metrics(y_true, y_pred):
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


class DistillDataset(Dataset):
    """Text + hard label + soft label (MoE teacher probs) cho từng sample."""
    def __init__(self, texts, hard_labels, teacher_probs, tokenizer, max_length=128):
        self.texts = texts
        self.hard_labels = hard_labels
        self.teacher_probs = teacher_probs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.hard_labels[idx], dtype=torch.long),
            "teacher_probs": torch.tensor(self.teacher_probs[idx], dtype=torch.float32),
        }


def distill_loss(student_logits, teacher_probs, hard_labels, temperature=4.0, alpha=0.7):
    """
    Loss = alpha * KD(student, teacher_probs) + (1-alpha) * CE(student, hard_labels)
    KD dùng KL divergence với temperature scaling (Hinton et al. 2015).
    """
    student_log_probs_T = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs_T = teacher_probs.clamp(min=1e-8)
    teacher_probs_T = teacher_probs_T ** (1.0 / temperature)
    teacher_probs_T = teacher_probs_T / teacher_probs_T.sum(dim=-1, keepdim=True)

    kd_loss = F.kl_div(student_log_probs_T, teacher_probs_T, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, hard_labels)

    return alpha * kd_loss + (1 - alpha) * ce_loss


class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best = 0, None

    def __call__(self, score):
        if self.best is None or score > self.best + self.min_delta:
            self.best, self.counter = score, 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]

    aug_dir = Path(config["data"].get("augmented_dir", "data/augmented"))
    dfs = {s: pd.read_csv(aug_dir / f"{s}.csv") for s in ["train", "val", "test"]}

    teacher_probs = {}
    teacher_labels = {}
    for s in ["train", "val", "test"]:
        teacher_probs[s] = np.load(os.path.join(args.moe_teacher_dir, f"{s}_probs.npy"))
        teacher_labels[s] = np.load(os.path.join(args.moe_teacher_dir, f"{s}_labels.npy"))
        df_labels = dfs[s][label_col].astype(int).values
        assert np.array_equal(df_labels, teacher_labels[s]), (
            f"Label mismatch giữa {s}.csv và {s}_labels.npy — kiểm tra lại "
            f"data/augmented có khớp với lúc train experts không."
        )

    datasets = {}
    for s in ["train", "val", "test"]:
        datasets[s] = DistillDataset(
            dfs[s][text_col].astype(str).tolist(),
            dfs[s][label_col].astype(int).tolist(),
            teacher_probs[s], tokenizer, max_length=args.max_seq_length,
        )

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size * 2, shuffle=False)

    console.print(f"[bold cyan]Building TinyPhoBERT student...[/bold cyan]")
    model = TinyPhoBERT(num_labels=3).to(device)
    console.print(f"  Student params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_steps = len(train_loader) * args.num_epochs
    num_warmup = int(num_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_steps)

    early_stopping = EarlyStopping(patience=args.patience)
    best_f1, best_epoch = 0.0, 0
    history = []

    console.print(f"\n[bold cyan]Distilling ({args.num_epochs} epochs max)...[/bold cyan]\n")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_p = batch["teacher_probs"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = distill_loss(outputs["logits"], teacher_p, labels, temperature=args.temperature, alpha=args.alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
                preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(batch["labels"].numpy().tolist())

        val_metrics = compute_metrics(all_labels, all_preds)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader), **{f"val_{k}": v for k, v in val_metrics.items()}})

        console.print(
            f"Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_f1_off={val_metrics.get('f1_offensive',0):.4f} | "
            f"val_f1_hate={val_metrics.get('f1_hate',0):.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            torch.save({"model_state_dict": model.state_dict(), "val_f1": best_f1, "epoch": epoch},
                       os.path.join(args.output_dir, "best_model.pt"))
            console.print(f"  [bold green]✓ Best model saved (Macro-F1={best_f1:.4f})[/bold green]")

        if early_stopping(val_metrics["macro_f1"]):
            console.print(f"\n[bold yellow]Early stopping tại epoch {epoch}.[/bold yellow]")
            break

    console.print(f"\n[bold green]Distillation complete! Best Val Macro-F1={best_f1:.4f} tại epoch {best_epoch}[/bold green]")
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())

    test_metrics = compute_metrics(all_labels, all_preds)
    table = Table(title="TinyPhoBERT (MoE-Distilled) Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    os.makedirs("results", exist_ok=True)
    with open("results/moe_distill_results.json", "w") as f:
        json.dump({**test_metrics, "best_epoch": best_epoch, "student_params": model.count_parameters()}, f, indent=2)
    console.print("Results saved to results/moe_distill_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe_teacher_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/teacher_config.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tinyphobert_moe_distilled")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7, help="Trọng số KD loss vs CE loss")
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(args, config)


if __name__ == "__main__":
    main()