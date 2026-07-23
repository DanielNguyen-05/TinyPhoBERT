"""
evaluation/save_all_probs.py

Mở rộng từ save_teacher_probs.py: lưu xác suất cho CẢ 3 splits
(train/val/test), không chỉ val/test — vì Gating Network cần xác suất
trên TRAIN set để học cách kết hợp các expert.

Hỗ trợ 2 loại model:
    - "teacher": PhoBERTTeacher-style (PhoBERT-large/base/v2, ViSoBERT, viBERT)
    - "qwen": Qwen2.5 QLoRA classifier (kiến trúc khác, cần load riêng)

Usage:
    # Cho model dạng PhoBERTTeacher (SupCon-Teacher, PhoBERT_v2+FGM, ViSoBERT, viBERT):
    python evaluation/save_all_probs.py --model_type teacher \\
        --checkpoint checkpoints/phobert_v2_fgm_noaug/best_model.pt \\
        --config configs/phobert_v2_fgm_noaug_config.yaml \\
        --output_dir checkpoints/phobert_v2_fgm_noaug

    # Cho Qwen (QLoRA):
    python evaluation/save_all_probs.py --model_type qwen \\
        --adapter_path checkpoints/qwen_classifier/best_adapter \\
        --config configs/qwen_config.yaml \\
        --output_dir checkpoints/qwen_classifier
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.seed import get_device

console = Console()


@torch.no_grad()
def extract_probs_teacher(model, dataloader, device):
    """Extract probs cho model kiểu PhoBERTTeacher (dict output với 'logits')."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(dataloader, desc="  Extracting", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs["logits"].float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


@torch.no_grad()
def extract_probs_qwen(model, dataloader, device):
    """Extract probs cho Qwen classifier (HuggingFace output với '.logits')."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(dataloader, desc="  Extracting", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def build_teacher_and_loaders(args, config, device):
    from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
    from utils.data_utils import HateSpeechDataset

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("config"):
        config = ckpt["config"]
        console.print("[cyan]Using the training config embedded in the checkpoint.[/cyan]")

    tokenizer = get_teacher_tokenizer(config["model"]["name"])
    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]
    max_len = config["training"]["max_seq_length"]

    aug_dir = Path(config["data"].get("augmented_dir", "data/augmented"))
    train_df = pd.read_csv(aug_dir / "train.csv")
    val_df = pd.read_csv(aug_dir / "val.csv")
    test_df = pd.read_csv(aug_dir / "test.csv")

    datasets = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        datasets[name] = HateSpeechDataset(
            df[text_col].astype(str).tolist(),
            df[label_col].astype(int).tolist(),
            tokenizer, max_length=max_len,
        )

    model = PhoBERTTeacher.from_pretrained_checkpoint(args.checkpoint).to(device)

    return model, datasets, extract_probs_teacher


def build_qwen_and_loaders(args, config, device):
    from models.qwen_classifier import load_qwen_with_adapter
    import torch.utils.data as tud

    class TextClsDataset(tud.Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length

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
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    model, tokenizer = load_qwen_with_adapter(config, args.adapter_path)
    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]
    max_len = config["model"].get("max_seq_length", 128)

    train_df = pd.read_csv(config["data"]["train_file"])
    val_df = pd.read_csv(config["data"]["val_file"])
    test_df = pd.read_csv(config["data"]["test_file"])

    datasets = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        datasets[name] = TextClsDataset(
            df[text_col].astype(str).tolist(),
            df[label_col].astype(int).tolist(),
            tokenizer, max_len,
        )

    return model, datasets, extract_probs_qwen


def main():
    parser = argparse.ArgumentParser(description="Extract probs cho train/val/test — dùng cho Gating Network.")
    parser.add_argument("--model_type", choices=["teacher", "qwen"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Cho model_type=teacher")
    parser.add_argument("--adapter_path", type=str, default=None, help="Cho model_type=qwen")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_type == "teacher":
        if not args.checkpoint:
            console.print("[bold red]✗ --checkpoint bắt buộc với model_type=teacher[/bold red]")
            return
        model, datasets, extract_fn = build_teacher_and_loaders(args, config, device)
    else:
        if not args.adapter_path:
            console.print("[bold red]✗ --adapter_path bắt buộc với model_type=qwen[/bold red]")
            return
        model, datasets, extract_fn = build_qwen_and_loaders(args, config, device)

    for split_name, dataset in datasets.items():
        console.print(f"\n[bold cyan]Extracting {split_name} probabilities...[/bold cyan]")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        probs, labels = extract_fn(model, loader, device)

        np.save(os.path.join(args.output_dir, f"{split_name}_probs.npy"), probs)
        np.save(os.path.join(args.output_dir, f"{split_name}_labels.npy"), labels)
        console.print(f"  Saved: {args.output_dir}/{split_name}_probs.npy | shape={probs.shape}")

    console.print("\n[bold green]Done![/bold green] Đủ train/val/test probs cho Gating Network.")


if __name__ == "__main__":
    main()
