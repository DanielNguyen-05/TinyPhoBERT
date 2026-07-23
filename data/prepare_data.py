"""
data/prepare_data.py

Chuẩn bị dữ liệu cho training: Preprocessing + Augmentation.

NGUYÊN TẮC QUAN TRỌNG:
    ✅ Preprocessing: áp dụng ĐỒNG NHẤT cho cả Train/Val/Test
       (vì đây là chuẩn hóa text, không tạo thông tin mới — phải nhất quán
        giữa lúc train và lúc inference/evaluate)

    ✅ Augmentation: CHỈ áp dụng cho Train
       (Val/Test phải giữ nguyên dữ liệu thật để đánh giá đúng khả năng
        tổng quát hóa của model. Augment val/test sẽ gây data leakage:
        bản gốc rơi vào train, bản augment gần giống rơi vào val/test
        → model "nhớ" thay vì "học" → metric bị thổi phồng giả tạo)

Pipeline:
    data/processed/{train,val,test}.csv   (đã split sẵn, CHƯA augment)
                    │
                    ▼
         [Preprocessing - cả 3 splits]
                    │
                    ▼
         [Augmentation - CHỈ train]
                    │
                    ▼
    data/augmented/{train,val,test}.csv   (sẵn sàng để train_teacher.py đọc)

Usage:
    python data/prepare_data.py
    python data/prepare_data.py --config configs/teacher_config.yaml
    python data/prepare_data.py --no_augment   # Chỉ preprocess, không augment
    python data/prepare_data.py --no_preprocess --no_augment  # Copy nguyên bản
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocess_vi import preprocess_dataframe
from utils.augment_eda import augment_minority_classes


def prepare_data(config: dict, apply_preprocess: bool = True, apply_augment: bool = True) -> None:
    data_cfg = config["data"]
    preprocess_cfg = config.get("preprocessing", {})
    aug_cfg = config.get("augmentation", {})

    text_col = data_cfg["text_col"]
    label_col = data_cfg["label_col"]

    # ── Load raw split data (đã chia sẵn train/val/test, CHƯA augment) ───────
    print("=" * 70)
    print("STEP 0: Loading raw split data")
    print("=" * 70)
    train_df = pd.read_csv(data_cfg["train_file"])
    val_df = pd.read_csv(data_cfg["val_file"])
    test_df = pd.read_csv(data_cfg["test_file"])

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = df[label_col].value_counts().to_dict()
        print(f"  {name}: {len(df):,} samples | dist={dist}")

    # ── STEP 1: Preprocessing — ĐỒNG NHẤT cho cả 3 splits ─────────────────────
    if apply_preprocess and preprocess_cfg.get("enabled", True):
        print("\n" + "=" * 70)
        print("STEP 1: Preprocessing (applied identically to train/val/test)")
        print("=" * 70)

        preprocess_kwargs = {
            "lowercase": preprocess_cfg.get("lowercase", True),
            "remove_url": preprocess_cfg.get("remove_url", True),
            "normalize_uni": preprocess_cfg.get("normalize_unicode", True),
            "reduce_repeat": preprocess_cfg.get("reduce_repeat", True),
            "apply_teen": preprocess_cfg.get("apply_teencode", True),
            "remove_emoji": preprocess_cfg.get("remove_emoji", False),
            "remove_phone": preprocess_cfg.get("remove_phone", True),
            "word_segment": preprocess_cfg.get("word_segment", False),
        }
        print(f"  Settings: {preprocess_kwargs}")

        train_df = preprocess_dataframe(train_df, text_col=text_col, output_col=text_col, **preprocess_kwargs)
        val_df = preprocess_dataframe(val_df, text_col=text_col, output_col=text_col, **preprocess_kwargs)
        test_df = preprocess_dataframe(test_df, text_col=text_col, output_col=text_col, **preprocess_kwargs)

        # Drop any rows that became empty after preprocessing
        for name, df_ref in [("train", "train_df"), ("val", "val_df"), ("test", "test_df")]:
            pass
        train_df = train_df[train_df[text_col].str.strip().astype(bool)].reset_index(drop=True)
        val_df = val_df[val_df[text_col].str.strip().astype(bool)].reset_index(drop=True)
        test_df = test_df[test_df[text_col].str.strip().astype(bool)].reset_index(drop=True)

        print(f"  ✓ train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,} (after dropping empties)")
    else:
        print("\n[Preprocessing] SKIPPED")

    # ── STEP 2: Augmentation — CHỈ trên TRAIN ─────────────────────────────────
    if apply_augment and aug_cfg.get("enabled", True):
        print("\n" + "=" * 70)
        print("STEP 2: EDA Augmentation (TRAIN ONLY — val/test untouched)")
        print("=" * 70)

        train_df = augment_minority_classes(
            train_df,
            text_col=text_col,
            label_col=label_col,
            target_classes=aug_cfg.get("target_classes", [1, 2]),
            augment_ratio=aug_cfg.get("augment_ratio", 0.5),
            num_aug_per_sample=aug_cfg.get("num_aug_per_sample", 3),
            alpha_sr=aug_cfg.get("alpha_sr", 0.1),
            alpha_ri=aug_cfg.get("alpha_ri", 0.1),
            alpha_rs=aug_cfg.get("alpha_rs", 0.1),
            p_rd=aug_cfg.get("p_rd", 0.1),
            random_state=data_cfg.get("seed", 42),
        )
    else:
        print("\n[Augmentation] SKIPPED")

    # ── STEP 3: Save to disk — data/augmented/ ────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Saving prepared data to disk")
    print("=" * 70)

    out_dir = Path(data_cfg.get("augmented_dir", "data/augmented"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  ✓ train → {train_path}  ({len(train_df):,} samples)")
    print(f"  ✓ val   → {val_path}  ({len(val_df):,} samples)  [untouched by augmentation]")
    print(f"  ✓ test  → {test_path}  ({len(test_df):,} samples)  [untouched by augmentation]")

    # ── Sanity check: NO overlap between augmented train and val/test ────────
    print("\n" + "=" * 70)
    print("STEP 4: Leakage sanity check")
    print("=" * 70)
    train_texts = set(train_df[text_col].astype(str).tolist())
    val_texts = set(val_df[text_col].astype(str).tolist())
    test_texts = set(test_df[text_col].astype(str).tolist())

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts

    print(f"  Exact-match overlap train∩val : {len(train_val_overlap)} texts")
    print(f"  Exact-match overlap train∩test: {len(train_test_overlap)} texts")
    if train_val_overlap or train_test_overlap:
        print("  [WARNING] Overlap detected — likely pre-existing duplicates in raw data,")
        print("            not caused by augmentation (augmentation only ran on train).")
    else:
        print("  ✓ No exact-match leakage between train and val/test.")

    print("\n[Done] Data preparation complete.")
    print(f"  Next step: point train_teacher.py to read from {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Prepare data: preprocessing + augmentation.")
    parser.add_argument("--config", type=str, default="configs/teacher_config.yaml")
    parser.add_argument("--no_preprocess", action="store_true", help="Skip preprocessing step.")
    parser.add_argument("--no_augment", action="store_true", help="Skip augmentation step.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    prepare_data(
        config,
        apply_preprocess=not args.no_preprocess,
        apply_augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
