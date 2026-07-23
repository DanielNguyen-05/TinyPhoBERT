"""
utils/__init__.py
"""
from .seed import set_seed, get_device
from .metrics import compute_metrics, huggingface_compute_metrics, LABEL_NAMES

__all__ = [
    "set_seed",
    "get_device",
    "compute_metrics",
    "huggingface_compute_metrics",
    "LABEL_NAMES",
    "HateSpeechDataset",
    "LABEL2ID",
    "ID2LABEL",
]


def __getattr__(name):
    # Keep lightweight utilities importable in probability-only evaluation
    # environments where Transformers is intentionally not installed.
    if name in {"HateSpeechDataset", "LABEL2ID", "ID2LABEL"}:
        from .data_utils import HateSpeechDataset, LABEL2ID, ID2LABEL

        return {
            "HateSpeechDataset": HateSpeechDataset,
            "LABEL2ID": LABEL2ID,
            "ID2LABEL": ID2LABEL,
        }[name]
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
