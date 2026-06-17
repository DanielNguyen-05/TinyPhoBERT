"""
data/download_data.py

Downloads and preprocesses the ViHSD dataset.

Dataset: Vietnamese Hate Speech Detection (ViHSD)
Source:  https://huggingface.co/datasets/visolex/ViHSD  (public, no auth needed)
         33,400 annotated Vietnamese social media comments
         Labels: 0=CLEAN, 1=OFFENSIVE, 2=HATE

Usage:
    python data/download_data.py                # Tự động (khuyến nghị)
    python data/download_data.py --source hf    # Via HuggingFace library
    python data/download_data.py --hf_token TOKEN  # Nếu cần token (gated datasets)
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}

# ── Dataset sources theo thứ tự ưu tiên ──────────────────────────────────────
# visolex/ViHSD: Public mirror, không cần auth, format chuẩn
# uitnlp/vihsd:  Official, nhưng gated (cần HF token)
HF_DATASET_IDS = [
    "visolex/ViHSD",    # ✅ Public, không cần auth — DÙNG CÁI NÀY
    "uitnlp/vihsd",     # Official (gated — cần HF_TOKEN)
]


def download_from_huggingface(hf_token: str = None) -> None:
    """
    Download ViHSD using the HuggingFace `datasets` library.
    Tự động split train/val/test theo cột 'type' nếu có.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] Cần cài: pip install datasets")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[Download] Tải ViHSD từ HuggingFace...")

    dataset = None
    for hf_id in HF_DATASET_IDS:
        try:
            print(f"  → Thử: {hf_id}")
            kwargs = {}
            if hf_token:
                kwargs["token"] = hf_token
            dataset = load_dataset(hf_id, **kwargs)
            print(f"  ✓ Tải thành công từ: {hf_id}")
            print(f"     {dataset}")
            break
        except Exception as e:
            print(f"  ✗ Thất bại ({hf_id}): {e}")
            continue

    if dataset is None:
        print("\n[Error] Không thể tải từ HuggingFace.")
        print("Hãy thử một trong các cách sau:")
        print("  1. Đăng nhập HuggingFace: huggingface-cli login")
        print("  2. Dùng token:  python data/download_data.py --hf_token YOUR_TOKEN")
        print("  3. Tải thủ công: https://huggingface.co/datasets/uitnlp/vihsd")
        print("     (Download train.csv, dev.csv, test.csv → đặt vào data/raw/)")
        sys.exit(1)

    # visolex/ViHSD có cột 'type' để phân loại train/dev/test trong 1 split
    # uitnlp/vihsd có thể có nhiều splits
    if "train" in dataset and len(dataset) == 1:
        # Single split (visolex/ViHSD): cần tách theo cột 'type'
        df = dataset["train"].to_pandas()
        print(f"\n  Dataset có {len(df):,} rows | Columns: {list(df.columns)}")

        if "type" in df.columns:
            # Split by 'type' column: train/dev/test
            for split_val in df["type"].unique():
                split_df = df[df["type"] == split_val].reset_index(drop=True)
                filename = f"{split_val}.csv"
                save_path = RAW_DIR / filename
                split_df.to_csv(save_path, index=False)
                print(f"  ✓ Saved {split_val}: {len(split_df):,} rows → {save_path}")
        else:
            # Không có cột 'type' → lưu toàn bộ là train
            save_path = RAW_DIR / "train.csv"
            df.to_csv(save_path, index=False)
            print(f"  ✓ Saved train: {len(df):,} rows → {save_path}")
    else:
        # Multiple splits (uitnlp/vihsd style)
        split_name_map = {"validation": "dev", "val": "dev"}
        for split_name, split_data in dataset.items():
            filename = split_name_map.get(split_name, split_name) + ".csv"
            df = split_data.to_pandas()
            save_path = RAW_DIR / filename
            df.to_csv(save_path, index=False)
            print(f"  ✓ Saved {split_name}: {len(df):,} rows → {save_path}")


def preprocess() -> None:
    """
    Chuẩn hoá và tạo data/processed/{train,val,test}.csv.

    Output format:
        free_text  | label_id | label_name
        "comment"  |    0     | CLEAN
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Map từ tên file raw → tên split output
    split_map = {
        "train": ["train.csv"],
        "val":   ["dev.csv", "val.csv", "validation.csv"],
        "test":  ["test.csv"],
    }

    print("\n[Preprocess] Chuẩn hoá dữ liệu...")
    all_dfs = {}

    for split, candidates in split_map.items():
        df = None
        for candidate in candidates:
            path = RAW_DIR / candidate
            if path.exists():
                df = pd.read_csv(path)
                print(f"  Loaded {path}: {len(df):,} rows | Columns: {list(df.columns)}")
                break
        if df is None:
            print(f"  [Info] Không tìm thấy file cho split '{split}' trong data/raw/")
            continue
        all_dfs[split] = df

    # Nếu không có val/test → chia 80/10/10 từ train
    if "val" not in all_dfs or "test" not in all_dfs:
        print("\n[Preprocess] Chia 80/10/10 từ train...")
        full_df = all_dfs.get("train")
        if full_df is None:
            print("[Error] Không tìm thấy dữ liệu train!")
            sys.exit(1)
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(full_df)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        all_dfs["train"] = full_df.iloc[:n_train]
        all_dfs["val"]   = full_df.iloc[n_train : n_train + n_val]
        all_dfs["test"]  = full_df.iloc[n_train + n_val :]
        print(f"  Train: {len(all_dfs['train']):,} | Val: {len(all_dfs['val']):,} | Test: {len(all_dfs['test']):,}")

    # ── Chuẩn hoá cột ──────────────────────────────────────────────────────
    LABEL_COLS = ["label_id", "label", "Label", "labels", "hate_speech_label"]
    TEXT_COLS  = ["free_text", "text", "comment", "sentence", "content"]

    for split, df in all_dfs.items():
        # Detect text column
        text_col = next((c for c in TEXT_COLS if c in df.columns), None)
        if text_col is None:
            text_col = df.columns[0]
            print(f"  [Warning] Dùng cột đầu làm text: '{text_col}'")

        # Detect label column
        label_col = next((c for c in LABEL_COLS if c in df.columns), None)
        if label_col is None:
            label_col = df.columns[-1]
            print(f"  [Warning] Dùng cột cuối làm label: '{label_col}'")

        out = pd.DataFrame()
        out["free_text"] = df[text_col].astype(str).str.strip()

        # Map labels → int
        sample_label = str(df[label_col].iloc[0]).strip()
        if sample_label.upper() in LABEL2ID:
            # String labels: CLEAN/OFFENSIVE/HATE
            out["label_id"]   = df[label_col].astype(str).str.strip().str.upper().map(LABEL2ID)
            out["label_name"] = df[label_col].astype(str).str.strip().str.upper()
        else:
            # Integer labels: 0/1/2
            out["label_id"]   = pd.to_numeric(df[label_col], errors="coerce")
            id2name = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
            out["label_name"] = out["label_id"].map(id2name)

        out = out.dropna(subset=["free_text", "label_id"])
        out["label_id"] = out["label_id"].astype(int)

        save_path = PROCESSED_DIR / f"{split}.csv"
        out.to_csv(save_path, index=False)

        dist = out["label_name"].value_counts().to_dict()
        print(f"  ✓ {split:5s}: {len(out):,} rows | {dist}")

    print("\n[Done] ✅ Dữ liệu đã sẵn sàng:")
    print("  → data/processed/train.csv")
    print("  → data/processed/val.csv")
    print("  → data/processed/test.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess ViHSD dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["auto", "hf"],
        default="auto",
        help="Nguồn tải: auto (mặc định), hf (HuggingFace library)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "HuggingFace API token (nếu cần cho gated datasets).\n"
            "Lấy tại: https://huggingface.co/settings/tokens\n"
            "Hoặc dùng: huggingface-cli login"
        ),
    )
    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Chỉ chạy bước preprocess (nếu đã có data/raw/)",
    )
    args = parser.parse_args()

    # Check if already processed
    if all((PROCESSED_DIR / f"{s}.csv").exists() for s in ["train", "val", "test"]):
        print("[Info] Đã có data/processed/. Dùng --preprocess_only để chạy lại preprocess.")
        counts = {s: len(pd.read_csv(PROCESSED_DIR / f"{s}.csv")) for s in ["train", "val", "test"]}
        for s, n in counts.items():
            print(f"  {s}: {n:,} rows")
        return

    if args.preprocess_only:
        preprocess()
        return

    download_from_huggingface(hf_token=args.hf_token)
    preprocess()


if __name__ == "__main__":
    main()
