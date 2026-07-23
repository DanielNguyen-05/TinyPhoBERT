"""
utils/preprocess_vi.py

Vietnamese Text Preprocessing for Hate Speech Detection.

Inspired by: "Vietnamese Hate and Offensive Detection using PhoBERT-CNN
and Social Media Streaming Data" (Tran et al., 2022 — arXiv:2206.00524)

Các bước xử lý:
    Phase 1 (Normalization):
        - Lowercase
        - Delete redundant spaces
        - Delete URLs/links
        - Normalize Unicode (NFC)
        - Delete redundant repeated characters (thichhhhh → thich)
        - Normalize accented Vietnamese letters
        - De-teencode: chuyển teen code → từ chuẩn
        - Emoji handling

    Phase 2 (Optional):
        - Remove stopwords
        - Keep offensive/hate keywords (không xóa những từ quan trọng)
"""

import re
import unicodedata
from typing import Optional


# ── Teen-code dictionary (Vietnamese social media) ────────────────────────────
# CHỈ giữ teen-code >= 2 ký tự để tránh false-positive match.
# Single-character entries (t→tôi, m→mày, n→nó, v→vậy, r→rồi, j→gì, z→vậy)
# đã bị LOẠI BỎ — chúng dễ match nhầm vì không có ngữ cảnh để phân biệt
# "t" đứng riêng là teen-code hay là phần còn sót lại từ lỗi tách từ khác.
TEENCODE_DICT = {
    # Insults / reactions (đặc trưng quan trọng cho hate speech detection)
    "vl": "vãi lồn",
    "vcl": "vãi cả lồn",
    "vkl": "vãi kl",
    "đmm": "đụ má mày",
    "đm": "đụ má",
    "dm": "đụ má",
    "clm": "cặc lồn mày",
    "cl": "cặc lồn",
    "loz": "lồn",
    "đcm": "đụ cái mồng",
    # Affirmatives / Negatives (>= 2 ký tự only)
    "ko": "không",
    "kg": "không",
    "kh": "không",
    "đc": "được",
    "dc": "được",
    "đk": "được",
    "dk": "được",
    "vs": "với",
    "ntn": "như thế nào",
    "nt": "nhắn tin",
    "ns": "nói sao",
    "nma": "nhưng mà",
    "nhma": "nhưng mà",
    "cx": "cũng",
    "cg": "cũng",
    "mk": "mình",
    "mn": "mọi người",
    "mng": "mọi người",
    "bn": "bạn",
    "bh": "bây giờ",
    "bjo": "bao giờ",
    "lm": "làm",
    "lun": "luôn",
    "luon": "luôn",
    "trc": "trước",
    "tg": "thời gian",
    "ms": "mới",
    "ck": "chồng",
    "vk": "vợ",
    "hj": "hì",
}

# Pattern: ký tự lặp >= 2 lần liên tiếp → rút về 1 ký tự duy nhất.
# LÝ DO rút về 1 thay vì 2: PhoBERT-large được pretrain trên text CHUẨN
# (Wikipedia, báo chí) — nó chưa từng thấy "vuill" (2 ký tự lặp) trong lúc
# pretrain, chỉ thấy "vui". Rút về 1 ký tự giúp khớp đúng với từ vựng mà
# model đã học, tận dụng được kiến thức pretrained thay vì tạo từ lạ.
_REPEAT_CHAR_PATTERN = re.compile(r'(.)\1{1,}')

# Ngoại lệ: một số từ lặp ký tự là cách viết chuẩn (không phải lỗi gõ),
# KHÔNG rút gọn các từ này
_REPEAT_EXCEPTIONS = {"hehe", "haha", "hihi", "huhu"}

# URL pattern
_URL_PATTERN = re.compile(
    r'https?://\S+|www\.\S+|bit\.ly/\S+|fb\.com/\S+|t\.co/\S+'
)

# Emoji pattern (basic Unicode ranges)
_EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F'  # Emoticons
    r'\U0001F300-\U0001F5FF'  # Symbols & Pictographs
    r'\U0001F680-\U0001F6FF'  # Transport & Map
    r'\U0001F1E0-\U0001F1FF'  # Flags
    r'\U00002702-\U000027B0'
    r'\U000024C2-\U0001F251]+',
    flags=re.UNICODE
)

# Multiple spaces
_MULTI_SPACE_PATTERN = re.compile(r'\s+')

# Phone numbers / sensitive info
_PHONE_PATTERN = re.compile(r'\b(0[0-9]{9,10})\b')


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form (chuẩn hóa Unicode)."""
    return unicodedata.normalize("NFC", text)


def remove_urls(text: str) -> str:
    """Remove URLs and links."""
    return _URL_PATTERN.sub(" ", text)


def remove_redundant_chars(text: str) -> str:
    """
    Rút ký tự lặp liên tiếp (>=2, kể cả khi ký tự đầu có dấu, các ký tự
    lặp theo sau không dấu — ví dụ 'quáaaaa') về còn 1 ký tự duy nhất
    mang dấu thanh gốc.
    Ví dụ: 'thichhhhh' → 'thich'
           'quáaaaa' → 'quá'   (giữ dấu sắc ở ký tự gốc)
           'ahhhhhh' → 'ah'
    Bảo toàn các từ trong _REPEAT_EXCEPTIONS (hehe, haha, hihi, huhu)
    vì đây là cách viết chuẩn, không phải lỗi gõ thừa ký tự.
    """
    # Map mỗi ký tự có dấu về base letter không dấu để so sánh "cùng nguyên âm"
    _BASE_MAP = {}
    _vowel_groups = {
        "a": "aàáảãạăằắẳẵặâầấẩẫậ",
        "e": "eèéẻẽẹêềếểễệ",
        "i": "iìíỉĩị",
        "o": "oòóỏõọôồốổỗộơờớởỡợ",
        "u": "uùúủũụưừứửữự",
        "y": "yỳýỷỹỵ",
    }
    for base, variants in _vowel_groups.items():
        for ch in variants:
            _BASE_MAP[ch] = base

    def _base(ch: str) -> str:
        return _BASE_MAP.get(ch.lower(), ch.lower())

    def reduce_word(w: str) -> str:
        if not w:
            return w
        result = [w[0]]
        for ch in w[1:]:
            prev = result[-1]
            same_vowel_group = ch.isalpha() and prev.isalpha() and _base(ch) == _base(prev)
            same_consonant = ch.isalpha() and prev.isalpha() and ch.lower() == prev.lower() and _base(ch) == ch.lower()
            if same_vowel_group or ch.lower() == prev.lower():
                continue  # skip repeated same character (vowel-group or exact consonant match)
            result.append(ch)
        return "".join(result)

    words = text.split()
    out = []
    for w in words:
        if w.lower() in _REPEAT_EXCEPTIONS:
            out.append(w)
        else:
            out.append(reduce_word(w))
    return " ".join(out)


def remove_emojis(text: str, replace_with: str = " ") -> str:
    """Remove or replace emojis."""
    return _EMOJI_PATTERN.sub(replace_with, text)


def apply_teencode(text: str, teencode_dict: dict = None) -> str:
    """
    Chuyển đổi teen code → từ chuẩn.
    Chỉ áp dụng với whole-word match để tránh false positives.
    """
    if teencode_dict is None:
        teencode_dict = TEENCODE_DICT

    words = text.split()
    result = []
    for w in words:
        # Try exact match (lowercase)
        w_lower = w.lower()
        if w_lower in teencode_dict:
            result.append(teencode_dict[w_lower])
        else:
            result.append(w)
    return " ".join(result)


def segment_vietnamese_words(text: str) -> str:
    """
    Segment Vietnamese words using underscores, as expected by PhoBERT.

    PhoBERT was pretrained on RDRSegmenter-style word-segmented text. Feeding
    raw whitespace-separated syllables creates a train/pretraining mismatch.
    Underthesea is used here because it is straightforward to run in the data
    preparation stage; segmentation is deliberately optional for models such
    as ViSoBERT that were pretrained on raw social-media text.
    """
    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise RuntimeError(
            "word_segment=true requires `underthesea`; install requirements.txt "
            "before preparing PhoBERT data."
        ) from exc
    return word_tokenize(text, format="text")


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_url: bool = True,
    normalize_uni: bool = True,
    reduce_repeat: bool = True,
    apply_teen: bool = True,
    remove_emoji: bool = False,    # False = giữ emoji (có thể mang sentiment)
    remove_phone: bool = True,
    word_segment: bool = False,
) -> str:
    """
    Pipeline tiền xử lý văn bản tiếng Việt cho hate speech detection.

    Thứ tự xử lý quan trọng:
    1. Unicode normalization (trước mọi thứ)
    2. Lowercase
    3. URL removal
    4. Emoji handling
    5. Phone anonymization
    6. Redundant char reduction
    7. Teen code conversion
    8. Whitespace cleanup

    Args:
        text: Raw Vietnamese text.
        lowercase: Convert to lowercase.
        remove_url: Remove URLs.
        normalize_uni: Normalize Unicode to NFC.
        reduce_repeat: Reduce repeated characters.
        apply_teen: Apply teen code dictionary.
        remove_emoji: Remove emojis (default False).
        remove_phone: Anonymize phone numbers.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode normalization
    if normalize_uni:
        text = normalize_unicode(text)

    # 2. Lowercase
    if lowercase:
        text = text.lower()

    # 3. Remove URLs
    if remove_url:
        text = remove_urls(text)

    # 4. Emoji handling
    if remove_emoji:
        text = remove_emojis(text, replace_with=" ")
    else:
        # Keep emoji but normalize spacing around them
        text = _EMOJI_PATTERN.sub(r' \g<0> ', text)

    # 5. Anonymize phone numbers
    if remove_phone:
        text = _PHONE_PATTERN.sub("[SĐT]", text)

    # 6. Reduce repeated characters (okkkk → ok)
    if reduce_repeat:
        text = remove_redundant_chars(text)

    # 7. Teen code conversion
    if apply_teen:
        text = apply_teencode(text)

    # 8. PhoBERT-compatible Vietnamese word segmentation (optional)
    if word_segment:
        text = segment_vietnamese_words(text)

    # 9. Final whitespace cleanup
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text


def preprocess_dataframe(
    df,
    text_col: str = "free_text",
    output_col: str = "free_text",
    **kwargs,
):
    """Apply preprocessing pipeline to a DataFrame column."""
    df = df.copy()
    df[output_col] = df[text_col].apply(lambda x: preprocess_text(x, **kwargs))
    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        "mày ngu vcllllll đm óc chó ko biết j hết",
        "https://fb.com/abc chó chết đi cho rảnh",
        "thichhhhh quaaaaa hihi k hiểu ns j z",
        "bình thường thôi mn ơi hehe",
        "lũ chó đói đcm dm mày nói gì vậy",
    ]
    print("=== Vietnamese Text Preprocessing Test ===\n")
    for ex in examples:
        cleaned = preprocess_text(ex)
        print(f"RAW    : {ex}")
        print(f"CLEAN  : {cleaned}")
        print()
