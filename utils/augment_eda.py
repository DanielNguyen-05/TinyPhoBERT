"""
utils/augment_eda.py

Easy Data Augmentation (EDA) for Vietnamese Hate Speech Detection.

Áp dụng từ:
    - Wei & Zou (2019): "EDA: Easy Data Augmentation Techniques for
      Boosting Performance on Text Classification Tasks"
    - Tran et al. (2022): arXiv:2206.00524 — EDA on ViHSD dataset

Các phép augment:
    1. SR  — Synonym Replacement (thay thế từ bằng synonym)
    2. RI  — Random Insertion (chèn synonym ngẫu nhiên)
    3. RS  — Random Swap (hoán đổi 2 từ ngẫu nhiên)
    4. RD  — Random Deletion (xóa từ ngẫu nhiên)
    5. BT  — Back-Translation (dịch VI→EN→VI, nếu có model)
    6. Mixup — Nội suy embedding-level (trong train_student.py)

Quan trọng:
    - Chỉ augment MINORITY classes: OFFENSIVE (label=1) và HATE (label=2)
    - KHÔNG augment CLEAN (label=0) — tránh label noise
    - Giữ nguyên offensive/hate keywords (không thay thế)
"""

import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ── Vietnamese Synonym Dictionary ─────────────────────────────────────────────
# Tập synonym cho các từ common (không phải offensive words)
# Offensive words KHÔNG được thay thế để giữ label tính chính xác
VI_SYNONYMS = {
    # Common adjectives
    "ngu": ["đần", "ngốc", "ngu ngốc", "dốt"],
    "đần": ["ngu", "ngốc", "dốt"],
    "xấu": ["tệ", "kém", "dở"],
    "tốt": ["hay", "giỏi", "tuyệt"],
    "đẹp": ["xinh", "dễ thương"],
    "xấu xí": ["tệ hại", "kém cỏi"],
    # Common verbs
    "nói": ["phát biểu", "kể", "bảo"],
    "làm": ["thực hiện", "tiến hành"],
    "đi": ["đến", "tới", "sang"],
    "biết": ["hiểu", "rõ"],
    "thấy": ["nhìn", "nghe"],
    "nghĩ": ["cho rằng", "tin", "tưởng"],
    # Common nouns
    "người": ["con người", "kẻ", "tên"],
    "mày": ["bạn", "anh", "chị", "em"],
    "tao": ["tôi", "mình", "ta"],
    "chúng": ["bọn", "lũ"],
    # Intensifiers
    "rất": ["cực kỳ", "vô cùng", "hết sức"],
    "quá": ["lắm", "thật sự"],
    "thật": ["thực sự", "thực ra"],
}

# Sensitive / offensive words — KHÔNG được thay thế trong SR
# (thay thế sẽ làm mất đặc trưng của hate speech)
_OFFENSIVE_KEYWORDS = {
    "chó", "lợn", "ngu", "đần", "ngốc", "chết", "giết", "vl", "vcl", "dm",
    "đm", "đmm", "cl", "loz", "lol", "đcm", "cc", "cặc", "lồn", "mẹ", "bố",
    "hại", "ghét", "căm", "thù", "tiêu diệt", "tắt thở", "xóa sổ",
    "óc chó", "óc lợn", "đồ", "thứ", "loài", "lũ", "bọn",
}


def _get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word. Returns empty list if none available."""
    word_lower = word.lower()
    # Don't replace offensive keywords
    if word_lower in _OFFENSIVE_KEYWORDS:
        return []
    return VI_SYNONYMS.get(word_lower, [])


def synonym_replacement(words: List[str], n: int) -> List[str]:
    """
    SR: Thay thế n từ ngẫu nhiên bằng synonym.
    Bỏ qua offensive keywords để giữ label tính.
    """
    new_words = words.copy()
    # Find replaceable words (have synonyms, not offensive)
    replaceable = [
        (i, w) for i, w in enumerate(words)
        if _get_synonyms(w) and w.lower() not in _OFFENSIVE_KEYWORDS
    ]

    if not replaceable:
        return new_words

    random.shuffle(replaceable)
    for i, (idx, word) in enumerate(replaceable[:n]):
        synonyms = _get_synonyms(word)
        if synonyms:
            new_words[idx] = random.choice(synonyms)

    return new_words


def random_insertion(words: List[str], n: int) -> List[str]:
    """
    RI: Chèn n synonym ngẫu nhiên vào vị trí ngẫu nhiên.
    """
    new_words = words.copy()
    for _ in range(n):
        # Find a word with synonyms
        candidates = [w for w in words if _get_synonyms(w)]
        if not candidates:
            break
        word = random.choice(candidates)
        synonyms = _get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, synonym)
    return new_words


def random_swap(words: List[str], n: int) -> List[str]:
    """
    RS: Hoán đổi vị trí 2 từ ngẫu nhiên n lần.
    """
    new_words = words.copy()
    if len(new_words) < 2:
        return new_words
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def random_deletion(words: List[str], p: float) -> List[str]:
    """
    RD: Xóa mỗi từ với xác suất p.
    Không xóa offensive keywords.
    Giữ ít nhất 1 từ.
    """
    if len(words) == 1:
        return words

    new_words = []
    for w in words:
        # Never delete offensive keywords
        if w.lower() in _OFFENSIVE_KEYWORDS:
            new_words.append(w)
        elif random.random() > p:
            new_words.append(w)

    if not new_words:
        return [random.choice(words)]
    return new_words


def eda_augment(
    text: str,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 4,
) -> List[str]:
    """
    Áp dụng EDA để tạo num_aug augmented samples từ 1 text.

    Args:
        text: Input text (đã preprocessed).
        alpha_sr: Tỷ lệ từ được thay synonym.
        alpha_ri: Tỷ lệ từ được chèn thêm.
        alpha_rs: Tỷ lệ từ được hoán đổi.
        p_rd: Xác suất xóa mỗi từ.
        num_aug: Số lượng augmented texts tạo ra.

    Returns:
        List of augmented texts (không bao gồm original).
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return [text] * num_aug

    n_sr = max(1, int(alpha_sr * n))
    n_ri = max(1, int(alpha_ri * n))
    n_rs = max(1, int(alpha_rs * n))

    augmented = []
    ops = ["sr", "ri", "rs", "rd"]

    for i in range(num_aug):
        op = ops[i % len(ops)]
        if op == "sr":
            new_words = synonym_replacement(words, n_sr)
        elif op == "ri":
            new_words = random_insertion(words, n_ri)
        elif op == "rs":
            new_words = random_swap(words, n_rs)
        else:  # rd
            new_words = random_deletion(words, p_rd)

        augmented_text = " ".join(new_words).strip()
        if augmented_text and augmented_text != text:
            augmented.append(augmented_text)
        else:
            # Fallback: random swap as safe operation
            new_words = random_swap(words, max(1, n // 5))
            augmented.append(" ".join(new_words))

    return augmented


def augment_minority_classes(
    df: pd.DataFrame,
    text_col: str = "free_text",
    label_col: str = "label_id",
    target_classes: List[int] = None,
    augment_ratio: float = 1.0,
    num_aug_per_sample: int = 3,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Augment minority classes trong DataFrame để giảm class imbalance.

    Strategy (từ survey paper arXiv:2502.08960):
        - Chỉ augment minority classes (OFFENSIVE=1, HATE=2)
        - augment_ratio: tỷ lệ samples augment so với target balance
        - Kết hợp nhiều EDA operations khác nhau

    Args:
        df: Training DataFrame.
        text_col: Text column name.
        label_col: Label column name.
        target_classes: Classes to augment (default: [1, 2] = OFFENSIVE, HATE).
        augment_ratio: Target ratio của minority vs majority. 1.0 = full balance.
                       0.5 = half balance (soft augmentation, khuyến nghị).
        num_aug_per_sample: Số augmented samples tạo ra cho mỗi original sample.
        random_state: Random seed.

    Returns:
        Augmented DataFrame (original + synthetic samples).
    """
    random.seed(random_state)
    np.random.seed(random_state)

    if target_classes is None:
        target_classes = [1, 2]  # OFFENSIVE, HATE

    label_counts = Counter(df[label_col].tolist())
    majority_count = max(label_counts.values())

    print(f"\n[EDA Augmentation] Original distribution:")
    for cls, cnt in sorted(label_counts.items()):
        label_name = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}.get(cls, str(cls))
        print(f"  Class {cls} ({label_name}): {cnt:,} samples")

    new_rows = []

    for cls in target_classes:
        if cls not in label_counts:
            continue

        cls_count = label_counts[cls]
        target_count = int(majority_count * augment_ratio)
        n_needed = max(0, target_count - cls_count)

        if n_needed == 0:
            print(f"  Class {cls}: No augmentation needed.")
            continue

        # Get all samples of this class
        cls_df = df[df[label_col] == cls].reset_index(drop=True)

        # Calculate how many augmented samples per original sample
        aug_per_sample = max(1, min(num_aug_per_sample, n_needed // max(len(cls_df), 1) + 1))

        print(f"  Class {cls}: {cls_count} → target {target_count} "
              f"(need +{n_needed}, aug_per_sample={aug_per_sample})")

        generated = 0
        iterations = 0
        max_iter = n_needed * 3  # Safety limit

        while generated < n_needed and iterations < max_iter:
            # Pick a random sample from this class
            sample = cls_df.sample(1).iloc[0]
            text = str(sample[text_col])

            # Generate augmented texts
            aug_texts = eda_augment(
                text,
                alpha_sr=alpha_sr,
                alpha_ri=alpha_ri,
                alpha_rs=alpha_rs,
                p_rd=p_rd,
                num_aug=aug_per_sample,
            )

            for aug_text in aug_texts:
                if generated >= n_needed:
                    break
                if aug_text.strip():  # Only add non-empty texts
                    new_row = sample.copy()
                    new_row[text_col] = aug_text
                    if "sample_id" in new_row.index:
                        new_row["sample_id"] = (
                            f"{sample['sample_id']}:aug:{generated:08d}"
                        )
                    new_rows.append(new_row)
                    generated += 1

            iterations += 1

        print(f"  Class {cls}: Generated {generated} augmented samples.")

    if not new_rows:
        print("[EDA] No augmented samples generated.")
        return df

    aug_df = pd.DataFrame(new_rows)
    result_df = pd.concat([df, aug_df], ignore_index=True)

    # Shuffle
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Final distribution
    new_counts = Counter(result_df[label_col].tolist())
    print(f"\n[EDA Augmentation] Augmented distribution:")
    for cls, cnt in sorted(new_counts.items()):
        label_name = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}.get(cls, str(cls))
        orig = label_counts.get(cls, 0)
        print(f"  Class {cls} ({label_name}): {orig:,} → {cnt:,} (+{cnt - orig:,})")

    return result_df


def augment_with_back_translation_placeholder(
    texts: List[str],
    labels: List[int],
    target_classes: List[int] = None,
) -> Tuple[List[str], List[int]]:
    """
    Placeholder cho back-translation augmentation.
    
    Để dùng back-translation thực tế, cần:
        pip install transformers sentencepiece
        Model: Helsinki-NLP/opus-mt-vi-en và Helsinki-NLP/opus-mt-en-vi
    
    Tham khảo: arXiv:2502.08960 Section II-A1 (Deep generative models)
    
    Ví dụ implement thực tế:
        from transformers import MarianMTModel, MarianTokenizer
        
        vi_to_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
        en_to_vi = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        
        def back_translate(text):
            # VI → EN
            en_text = translate(vi_to_en, text)
            # EN → VI
            vi_text = translate(en_to_vi, en_text)
            return vi_text
    """
    print("[BackTranslation] Placeholder — implement với Helsinki-NLP models")
    return texts, labels


if __name__ == "__main__":
    # Quick test
    print("=== EDA Augmentation Test ===\n")

    test_texts = [
        "mày ngu vcl đm óc chó ko biết gì hết",
        "lũ chó đói đcm dm mày nói gì vậy",
        "thứ đồ vô học tầm thường như mày",
    ]
    test_labels = [2, 2, 1]  # HATE, HATE, OFFENSIVE

    for text, label in zip(test_texts, test_labels):
        label_name = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}[label]
        print(f"ORIGINAL [{label_name}]: {text}")
        augmented = eda_augment(text, num_aug=3)
        for i, aug in enumerate(augmented, 1):
            print(f"  AUG {i}: {aug}")
        print()
