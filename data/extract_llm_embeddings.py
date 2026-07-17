"""
data/extract_llm_embeddings.py

Trích xuất embedding từ Qwen2.5-0.5B (frozen) cho toàn bộ dataset.
Chạy 1 lần duy nhất, lưu .npy để train_fusion.py đọc lại — tránh
forward LLM mỗi epoch (tiết kiệm VRAM và thời gian đáng kể).

Qwen2.5-0.5B hidden size: 896-dim
Pooling strategy: mean-pool over non-padding tokens của last hidden state

Vì sao mean-pool thay vì last token (như LLM thường dùng)?
    - PhoBERT dùng CLS token (first) → mean-pool của Qwen tạo ra
      góc nhìn bổ sung thay vì cạnh tranh cùng cách aggregate.
    - Mean-pool robust hơn với câu ngắn (ViHSD avg 11 từ): last token
      của câu ngắn dễ bị dominated bởi punctuation/stopword.

Usage:
    python data/extract_llm_embeddings.py
    python data/extract_llm_embeddings.py --config configs/fusion_config.yaml
    python data/extract_llm_embeddings.py --data_dir data/augmented --batch_size 64
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def mean_pool_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool last hidden state, bỏ qua padding tokens.

    Args:
        last_hidden_state: (B, T, H)
        attention_mask: (B, T) — 1 = real token, 0 = padding

    Returns:
        (B, H) mean-pooled embedding
    """
    # Expand mask để nhân với hidden state
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)  # (B, H)
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)              # (B, 1)
    return sum_hidden / count  # (B, H)


@torch.no_grad()
def extract_embeddings(
    texts: list,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract mean-pooled embeddings từ LLM cho tất cả texts.

    Returns:
        numpy array shape (N, hidden_size)
    """
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Extracting"):
        batch_texts = texts[i : i + batch_size]

        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        pooled = mean_pool_hidden(last_hidden, attention_mask)  # (B, H)
        all_embeddings.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract LLM embeddings offline.")
    parser.add_argument("--config", type=str, default="configs/fusion_config.yaml")
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override data directory (default: from config)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID for the LLM extractor"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/llm_embeddings",
        help="Directory to save .npy embedding files"
    )
    args = parser.parse_args()

    # Load config
    if os.path.isfile(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        llm_cfg = config.get("llm_extractor", {})
        model_name = llm_cfg.get("model_name", args.model_name)
        output_dir = llm_cfg.get("embedding_dir", args.output_dir)
        max_length = llm_cfg.get("max_length", 128)
        data_cfg = config.get("data", {})
    else:
        model_name = args.model_name
        output_dir = args.output_dir
        max_length = 128
        data_cfg = {}

    # Determine data directory
    data_dir = args.data_dir or data_cfg.get("augmented_dir", "data/augmented")
    text_col = data_cfg.get("text_col", "free_text")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model và tokenizer
    print(f"\n[LLM] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    # In hidden size để xác nhận
    hidden_size = model.config.hidden_size
    print(f"  Hidden size: {hidden_size}-dim")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Output] Embeddings will be saved to: {output_dir}/")

    # Process each split
    splits = {
        "train": os.path.join(data_dir, "train.csv"),
        "val":   os.path.join(data_dir, "val.csv"),
        "test":  os.path.join(data_dir, "test.csv"),
    }

    for split_name, csv_path in splits.items():
        if not os.path.isfile(csv_path):
            print(f"  [Skip] {split_name}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).tolist()
        print(f"\n[{split_name}] {len(texts):,} samples → extracting...")

        embeddings = extract_embeddings(
            texts, model, tokenizer, device,
            batch_size=args.batch_size,
            max_length=max_length,
        )

        out_path = os.path.join(output_dir, f"{split_name}_llm_embeddings.npy")
        np.save(out_path, embeddings)
        print(f"  Saved: {out_path} | shape={embeddings.shape} | dtype={embeddings.dtype}")

    print(f"\n[Done] All embeddings saved to {output_dir}/")
    print("Next step: python training/train_fusion.py --config configs/fusion_config.yaml")


if __name__ == "__main__":
    main()