"""
evaluation/save_teacher_probs.py  [v2 - có chẩn đoán load_state_dict]
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

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from utils.data_utils import HateSpeechDataset
from utils.seed import get_device

console = Console()


@torch.no_grad()
def extract_probs(model, dataloader, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/teacher_large/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/teacher_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("config"):
        config = ckpt["config"]
        console.print("[cyan]Using the training config embedded in the checkpoint.[/cyan]")

    console.print("[bold cyan]Loading tokenizer and data...[/bold cyan]")
    tokenizer = get_teacher_tokenizer(config["model"]["name"])

    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]
    max_len = config["training"]["max_seq_length"]

    # QUAN TRỌNG: đọc từ augmented_dir (đã preprocess), KHÔNG đọc từ
    # config["data"]["val_file"]/["test_file"] (dữ liệu THÔ) — phải khớp
    # chính xác với cách train_teacher.py đọc dữ liệu lúc train, nếu không
    # model bị đánh giá trên text nó chưa từng thấy → điểm số tụt giả tạo.
    from pathlib import Path
    aug_dir = Path(config["data"].get("augmented_dir", "data/augmented"))
    val_df = pd.read_csv(aug_dir / "val.csv")
    test_df = pd.read_csv(aug_dir / "test.csv")

    val_ds = HateSpeechDataset(val_df[text_col].astype(str).tolist(), val_df[label_col].astype(int).tolist(), tokenizer, max_length=max_len)
    test_ds = HateSpeechDataset(test_df[text_col].astype(str).tolist(), test_df[label_col].astype(int).tolist(), tokenizer, max_length=max_len)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    console.print(f"[bold cyan]Loading Teacher checkpoint: {args.checkpoint}[/bold cyan]")
    console.print(f"  model_name trong config: [yellow]{config['model']['name']}[/yellow]")

    model = PhoBERTTeacher.from_pretrained_checkpoint(args.checkpoint)
    console.print(f"\n[bold]=== LOAD_STATE_DICT DIAGNOSTIC ===[/bold]")
    console.print(f"  Reconstructed head: {model.classification_head}")
    console.print("[bold green]  ✓ Architecture reconstructed from checkpoint config.[/bold green]\n")

    if "config" in ckpt:
        ckpt_model_name = ckpt["config"].get("model", {}).get("name", "?")
        console.print(f"  model_name LÚC TRAIN (lưu trong checkpoint): [yellow]{ckpt_model_name}[/yellow]")
        if ckpt_model_name != config["model"]["name"]:
            console.print(
                f"[bold red]✗ MISMATCH: config hiện tại dùng '{config['model']['name']}' "
                f"nhưng checkpoint được train với '{ckpt_model_name}'![/bold red]"
            )
    console.print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')} | val_f1 lưu trong ckpt: {ckpt.get('val_f1', '?')}\n")

    model = model.to(device)

    console.print("\n[bold cyan]Extracting Val probabilities...[/bold cyan]")
    val_probs, val_labels = extract_probs(model, val_loader, device)
    np.save(os.path.join(output_dir, "val_probs.npy"), val_probs)
    np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)
    console.print(f"  Saved: {output_dir}/val_probs.npy | shape={val_probs.shape}")

    console.print("\n[bold cyan]Extracting Test probabilities...[/bold cyan]")
    test_probs, test_labels = extract_probs(model, test_loader, device)
    np.save(os.path.join(output_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)
    console.print(f"  Saved: {output_dir}/test_probs.npy | shape={test_probs.shape}")

    console.print("\n[bold green]Done![/bold green] Kiểm tra lại phần DIAGNOSTIC ở trên trước khi chạy ensemble.")


if __name__ == "__main__":
    main()
