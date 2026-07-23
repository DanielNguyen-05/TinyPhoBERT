"""
training/train_qwen_classifier.py

QLoRA fine-tuning Qwen2.5 cho Vietnamese Hate Speech Detection.

Usage:
    python training/train_qwen_classifier.py --config configs/qwen_config.yaml
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen_classifier import build_qwen_classifier, load_qwen_with_adapter
from utils.data_utils import get_class_weights, get_weighted_sampler
from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device

console = Console()


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def focal_loss(logits, labels, weight=None, gamma=2.0, label_smoothing=0.1):
    with torch.no_grad():
        pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
        focal_w = (1.0 - pt) ** gamma
    ce = F.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing, reduction="none")
    return (focal_w * ce).mean()


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights,
                 focal_gamma, label_smoothing, grad_accum_steps=1, max_grad_norm=0.3):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = focal_loss(logits, labels, weight=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
        loss = loss / grad_accum_steps
        loss.backward()

        if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(dataloader)):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, class_weights, focal_gamma, label_smoothing, split_name="Val"):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(dataloader, desc=f"  {split_name}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = focal_loss(logits, labels, weight=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
        total_loss += loss.item()

        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs)

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics, all_labels, all_preds, np.concatenate(all_probs, axis=0)


def train(config: dict) -> None:
    training_cfg = config["training"]
    data_cfg = config["data"]

    set_seed(data_cfg["seed"])
    device = get_device()
    os.makedirs(training_cfg["output_dir"], exist_ok=True)

    console.print("=" * 70)
    console.print(f"[bold cyan]Qwen QLoRA Fine-tuning — {config['model']['name']}[/bold cyan]")
    console.print("=" * 70)

    model, tokenizer = build_qwen_classifier(config)
    if not config.get("quantization", {}).get("load_in_4bit", True):
        model = model.to(device)

    text_col = data_cfg["text_col"]
    label_col = data_cfg["label_col"]

    train_df = pd.read_csv(data_cfg["train_file"])
    val_df = pd.read_csv(data_cfg["val_file"])
    test_df = pd.read_csv(data_cfg["test_file"])

    train_texts = train_df[text_col].astype(str).tolist()
    train_labels = train_df[label_col].astype(int).tolist()
    val_texts = val_df[text_col].astype(str).tolist()
    val_labels = val_df[label_col].astype(int).tolist()
    test_texts = test_df[text_col].astype(str).tolist()
    test_labels = test_df[label_col].astype(int).tolist()

    console.print(f"  train: {len(train_texts):,} | val: {len(val_texts):,} | test: {len(test_texts):,}")

    max_len = config["model"].get("max_seq_length", 128)
    train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = TextClassificationDataset(val_texts, val_labels, tokenizer, max_len)
    test_ds = TextClassificationDataset(test_texts, test_labels, tokenizer, max_len)

    class_weights = get_class_weights(train_labels, num_classes=config["model"]["num_labels"]).to(device)

    use_sampler = training_cfg.get("use_weighted_sampler", True)
    train_loader = DataLoader(
        train_ds, batch_size=training_cfg["batch_size"],
        sampler=get_weighted_sampler(train_labels, strength=training_cfg.get("sampler_strength", 0.5)) if use_sampler else None,
        shuffle=not use_sampler,
        num_workers=training_cfg.get("dataloader_num_workers", 2),
    )
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"] * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=training_cfg["batch_size"] * 2, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"])

    grad_accum = training_cfg.get("gradient_accumulation_steps", 1)
    num_steps = math.ceil(len(train_loader) / grad_accum) * training_cfg["num_epochs"]
    num_warmup = int(num_steps * training_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_steps)
    console.print(f"  Schedule: Cosine | total_steps={num_steps} | warmup={num_warmup}")

    patience = training_cfg.get("early_stopping_patience", 5)
    early_stopping = EarlyStopping(patience=patience)
    console.print(f"  Early stopping: patience={patience}")

    best_f1 = 0.0
    best_epoch = 0
    history = []
    adapter_dir = os.path.join(training_cfg["output_dir"], "best_adapter")

    console.print(f"\n[bold cyan]Training ({training_cfg['num_epochs']} epochs max)...[/bold cyan]\n")

    for epoch in range(1, training_cfg["num_epochs"] + 1):
        console.print(f"[bold]Epoch {epoch}/{training_cfg['num_epochs']}[/bold]")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            class_weights, training_cfg.get("focal_gamma", 2.0),
            training_cfg.get("label_smoothing", 0.1),
            grad_accum_steps=grad_accum, max_grad_norm=training_cfg.get("max_grad_norm", 0.3),
        )
        val_metrics, _, _, _ = evaluate(
            model, val_loader, device, class_weights,
            training_cfg.get("focal_gamma", 2.0), training_cfg.get("label_smoothing", 0.1), "Val",
        )

        f1_off = val_metrics.get("f1_offensive", 0)
        f1_hate = val_metrics.get("f1_hate", 0)

        console.print(
            f"  Train: loss={train_metrics['loss']:.4f} | f1_macro={train_metrics['macro_f1']:.4f}\n"
            f"  Val  : loss={val_metrics['loss']:.4f}   | f1_macro={val_metrics['macro_f1']:.4f}\n"
            f"         f1_off={f1_off:.4f} | f1_hate={f1_hate:.4f}"
        )
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(adapter_dir)
            console.print(
                f"  [bold green]✓ Best adapter saved "
                f"(Macro-F1={best_f1:.4f} | F1_OFF={f1_off:.4f} | F1_HATE={f1_hate:.4f})[/bold green]"
            )

        if early_stopping(val_metrics["macro_f1"]):
            console.print(f"\n[bold yellow]Early stopping at epoch {epoch}.[/bold yellow]")
            break

    console.print(f"\n[bold green]Training complete! Best Val Macro-F1={best_f1:.4f} at epoch {best_epoch}[/bold green]")
    with open(os.path.join(training_cfg["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    console.print("\n[bold cyan]Rebuilding model + loading best adapter for final eval...[/bold cyan]")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, tokenizer = load_qwen_with_adapter(config, adapter_dir)
    if not config.get("quantization", {}).get("load_in_4bit", True):
        model = model.to(device)

    val_metrics_final, val_labels_out, val_preds_out, val_probs = evaluate(
        model, val_loader, device, class_weights,
        training_cfg.get("focal_gamma", 2.0), training_cfg.get("label_smoothing", 0.1), "Val-Final",
    )
    np.save(os.path.join(training_cfg["output_dir"], "val_probs.npy"), val_probs)
    np.save(os.path.join(training_cfg["output_dir"], "val_labels.npy"), np.array(val_labels_out))

    test_metrics, test_labels_out, test_preds, test_probs = evaluate(
        model, test_loader, device, class_weights,
        training_cfg.get("focal_gamma", 2.0), training_cfg.get("label_smoothing", 0.1), "Test",
    )
    np.save(os.path.join(training_cfg["output_dir"], "test_probs.npy"), test_probs)
    np.save(os.path.join(training_cfg["output_dir"], "test_labels.npy"), np.array(test_labels_out))

    console.print("\n[bold]Test Results:[/bold]")
    print_classification_report(test_labels_out, test_preds)

    table = Table(title=f"Qwen Classifier ({config['model']['name']}) Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    os.makedirs("results", exist_ok=True)
    with open("results/qwen_classifier_results.json", "w") as f:
        json.dump({**test_metrics, "best_epoch": best_epoch, "model": config["model"]["name"]}, f, indent=2)
    console.print("\nResults saved to results/qwen_classifier_results.json")
    console.print(f"Probabilities saved to {training_cfg['output_dir']}/{{val,test}}_probs.npy (dùng cho ensemble)")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5 for Vietnamese HSD.")
    parser.add_argument("--config", type=str, default="configs/qwen_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
