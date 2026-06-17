"""
data/download_data.py

Downloads and preprocesses the ViHSD dataset.

Usage:
    python data/download_data.py
    python data/download_data.py --source huggingface
    python data/download_data.py --source github
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}


def download_from_huggingface() -> None:
    """Download ViHSD from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install: pip install datasets")

    print("[Download] Loading ViHSD from HuggingFace...")
    # The dataset may be under different identifiers; try common ones
    dataset = None
    hf_ids = [
        "uet-ai/ViHSD",
        "phongnt109/vihsd",
        "datasets/ViHSD",
    ]
    for hf_id in hf_ids:
        try:
            dataset = load_dataset(hf_id)
            print(f"[Download] Loaded from: {hf_id}")
            break
        except Exception:
            continue

    if dataset is None:
        print("[Warning] Could not load from HuggingFace. Falling back to manual download.")
        download_manual()
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()
        save_path = RAW_DIR / f"{split_name}.csv"
        df.to_csv(save_path, index=False)
        print(f"[Download] Saved {split_name}: {len(df)} rows → {save_path}")

    preprocess()


def download_manual() -> None:
    """Download ViHSD from GitHub manually."""
    import urllib.request

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/sonlam1102/vihsd/main/data"
    files = {
        "train.csv": f"{base_url}/train.csv",
        "dev.csv": f"{base_url}/dev.csv",
        "test.csv": f"{base_url}/test.csv",
    }

    for filename, url in files.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"[Skip] {filename} already exists.")
            continue
        print(f"[Download] {url} → {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"[Error] Failed to download {filename}: {e}")
            print("Please download ViHSD manually from: https://github.com/sonlam1102/vihsd")
            sys.exit(1)

    preprocess()


def preprocess() -> None:
    """
    Standardize column names and create 80/10/10 splits if needed.
    Saves to data/processed/{train,val,test}.csv with columns:
        free_text  : original comment text
        label_id   : integer label (0=CLEAN, 1=OFFENSIVE, 2=HATE)
        label_name : string label
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Possible source files
    split_map = {
        "train": ["train.csv"],
        "val": ["dev.csv", "val.csv", "validation.csv"],
        "test": ["test.csv"],
    }

    print("\n[Preprocess] Standardizing columns...")
    all_dfs = {}
    for split, candidates in split_map.items():
        df = None
        for candidate in candidates:
            path = RAW_DIR / candidate
            if path.exists():
                df = pd.read_csv(path)
                print(f"  Loaded {path}: {len(df)} rows | Columns: {list(df.columns)}")
                break
        if df is None:
            print(f"  [Warning] No raw file found for split '{split}'")
            continue
        all_dfs[split] = df

    # If only train found, do 80/10/10 split
    if "val" not in all_dfs or "test" not in all_dfs:
        print("[Preprocess] No val/test found. Creating 80/10/10 split from train...")
        full_df = all_dfs.get("train")
        if full_df is None:
            print("[Error] No training data found at all!")
            sys.exit(1)
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(full_df)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        all_dfs["train"] = full_df.iloc[:n_train]
        all_dfs["val"] = full_df.iloc[n_train:n_train + n_val]
        all_dfs["test"] = full_df.iloc[n_train + n_val:]

    # Standardize columns
    LABEL_COLS = ["label", "Label", "label_id", "labels", "hate_speech_label"]
    TEXT_COLS = ["free_text", "text", "comment", "sentence", "content"]

    for split, df in all_dfs.items():
        # Detect text column
        text_col = next((c for c in TEXT_COLS if c in df.columns), None)
        if text_col is None:
            text_col = df.columns[0]
            print(f"  [Warning] Could not detect text column; using '{text_col}'")

        # Detect label column
        label_col = next((c for c in LABEL_COLS if c in df.columns), None)
        if label_col is None:
            label_col = df.columns[1]
            print(f"  [Warning] Could not detect label column; using '{label_col}'")

        out = pd.DataFrame()
        out["free_text"] = df[text_col].astype(str)

        # Map string labels to int if needed
        sample_label = df[label_col].iloc[0]
        if isinstance(sample_label, str) and sample_label.upper() in LABEL2ID:
            out["label_id"] = df[label_col].str.upper().map(LABEL2ID)
            out["label_name"] = df[label_col].str.upper()
        else:
            out["label_id"] = df[label_col].astype(int)
            id2name = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
            out["label_name"] = out["label_id"].map(id2name)

        # Drop rows with NaN
        out = out.dropna()

        save_path = PROCESSED_DIR / f"{split}.csv"
        out.to_csv(save_path, index=False)
        label_dist = out["label_name"].value_counts().to_dict()
        print(f"  Saved {split}: {len(out)} rows | Labels: {label_dist} → {save_path}")

    print("\n[Preprocess] Done! Files saved to data/processed/")
    print("  → data/processed/train.csv")
    print("  → data/processed/val.csv")
    print("  → data/processed/test.csv")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess ViHSD dataset.")
    parser.add_argument(
        "--source",
        choices=["huggingface", "github"],
        default="github",
        help="Data source (default: github)",
    )
    args = parser.parse_args()

    if args.source == "huggingface":
        download_from_huggingface()
    else:
        download_manual()


if __name__ == "__main__":
    main()
