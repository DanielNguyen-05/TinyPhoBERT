"""
utils/data_utils.py
Dataset loading and preprocessing for ViHSD.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# ViHSD label mapping
LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class HateSpeechDataset(Dataset):
    """
    PyTorch Dataset for Vietnamese Hate Speech Detection.

    Args:
        texts: List of input texts.
        labels: List of integer labels (0=CLEAN, 1=OFFENSIVE, 2=HATE).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_vihsd_from_csv(
    train_path: str,
    val_path: str,
    test_path: str,
    text_col: str = "free_text",
    label_col: str = "label_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ViHSD splits from CSV files."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        assert text_col in df.columns, f"Column '{text_col}' not found in {name}"
        assert label_col in df.columns, f"Column '{label_col}' not found in {name}"
        print(f"[Data] {name}: {len(df)} samples | Label dist: {df[label_col].value_counts().to_dict()}")

    return train_df, val_df, test_df


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    text_col: str = "free_text",
    label_col: str = "label_id",
    max_length: int = 128,
) -> Tuple[HateSpeechDataset, HateSpeechDataset, HateSpeechDataset]:
    """Build HateSpeechDataset objects from DataFrames."""
    train_texts = train_df[text_col].astype(str).tolist()
    train_labels = train_df[label_col].astype(int).tolist()

    val_texts = val_df[text_col].astype(str).tolist()
    val_labels = val_df[label_col].astype(int).tolist()

    test_texts = test_df[text_col].astype(str).tolist()
    test_labels = test_df[label_col].astype(int).tolist()

    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


def get_class_weights(
    labels: List[int],
    num_classes: int = 3,
) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced datasets.
    Useful for weighted cross-entropy loss.
    """
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes  # normalize
    return weights
