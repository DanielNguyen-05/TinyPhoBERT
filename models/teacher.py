"""
models/teacher.py

Teacher Model for Vietnamese Hate Speech Detection.

BẢN KHÔI PHỤC — đây là phiên bản KHÔNG có SupCon, đạt Macro-F1=65.38%
(accuracy=87.28%, best_epoch=22) trước khi SupCon được thêm vào gây NaN
và giảm hiệu suất xuống 61-63%.

Hỗ trợ cả PhoBERT-base (12L/768H) và PhoBERT-large (24L/1024H).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaConfig,
    RobertaModel,
)


class PhoBERTTeacher(nn.Module):
    """
    Teacher model: PhoBERT-large fine-tuned for hate speech classification.

    Architecture:
        PhoBERT-large (24 layers, 1024 hidden, 16 heads)
        └── Dropout
        └── Linear(1024 → 3)

    Args:
        model_name: HuggingFace model identifier (default: vinai/phobert-large).
        num_labels: Number of output classes (default: 3).
        dropout: Dropout probability for classifier head.
        output_hidden_states: Whether to return all hidden states.
        output_attentions: Whether to return all attention maps.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-large",
        num_labels: int = 3,
        dropout: float = 0.1,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # Load backbone (auto-detect base vs large) — KHÔNG override dropout
        # của backbone, giữ nguyên config gốc từ HuggingFace pretrained.
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Store config info — auto-detected from model
        self.hidden_size = self.config.hidden_size        # 768 (base) | 1024 (large)
        self.num_layers  = self.config.num_hidden_layers  # 12  (base) | 24   (large)
        self.num_heads   = self.config.num_attention_heads # 12 (base) | 16   (large)
        print(
            f"[Teacher] {model_name} | "
            f"{self.num_layers}L / {self.hidden_size}H / {self.num_heads}heads"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dictionary with:
                logits          : (B, num_labels)
                hidden_states   : tuple of (B, T, H) for each layer
                attentions      : tuple of (B, num_heads, T, T) for each layer
                loss            : scalar Focal/CE loss (if labels provided)
                pooled_output   : (B, H) CLS representation
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        # [CLS] token representation
        sequence_output = outputs.last_hidden_state  # (B, T, H)
        pooled_output = sequence_output[:, 0, :]     # (B, H)

        pooled_output_dropped = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_dropped)      # (B, 3)

        result = {
            "logits": logits,
            "pooled_output": pooled_output,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

        if labels is not None:
            result["loss"] = self._compute_loss(logits, labels)

        return result

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss hoặc Weighted CE tuỳ config."""
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(logits.device)

        if self.use_focal_loss:
            # Focal Loss: FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)
            ce = F.cross_entropy(
                logits, labels,
                weight=weight,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
            focal_w = (1.0 - pt) ** self.focal_gamma
            return (focal_w * ce).mean()
        else:
            return F.cross_entropy(
                logits, labels,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str = "vinai/phobert-large",
        num_labels: int = 3,
        **kwargs,
    ) -> "PhoBERTTeacher":
        """Load a fine-tuned teacher from a local checkpoint."""
        model = cls(model_name=model_name, num_labels=num_labels, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"[Teacher] Loaded checkpoint from: {checkpoint_path}")
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        """Freeze all parameters (use during distillation)."""
        for param in self.parameters():
            param.requires_grad = False
        print("[Teacher] All parameters frozen.")

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Teacher] All parameters unfrozen.")


def get_teacher_tokenizer(model_name: str = "vinai/phobert-large") -> AutoTokenizer:
    """Load PhoBERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[Teacher] Tokenizer loaded: {model_name} | Vocab size: {tokenizer.vocab_size}")
    return tokenizer