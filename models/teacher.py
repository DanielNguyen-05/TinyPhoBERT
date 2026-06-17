"""
models/teacher.py

PhoBERT-base Teacher Model for Vietnamese Hate Speech Detection.

Wraps vinai/phobert-base (RoBERTa architecture) with a classification head.
Provides interface to extract hidden states and attention maps for distillation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaConfig,
    RobertaModel,
)


class PhoBERTTeacher(nn.Module):
    """
    Teacher model: PhoBERT-base fine-tuned for hate speech classification.

    Architecture:
        PhoBERT-base (12 layers, 768 hidden, 12 heads)
        └── Dropout
        └── Linear(768 → 3)

    Args:
        model_name: HuggingFace model identifier (default: vinai/phobert-base).
        num_labels: Number of output classes (default: 3).
        dropout: Dropout probability for classifier head.
        output_hidden_states: Whether to return all hidden states.
        output_attentions: Whether to return all attention maps.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 3,
        dropout: float = 0.1,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels

        # Load PhoBERT backbone
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Store config info for distillation
        self.hidden_size = self.config.hidden_size       # 768
        self.num_layers = self.config.num_hidden_layers  # 12
        self.num_heads = self.config.num_attention_heads # 12

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
                hidden_states   : tuple of (B, T, 768) for each layer
                attentions      : tuple of (B, 12, T, T) for each layer
                loss            : scalar cross-entropy loss (if labels provided)
                pooled_output   : (B, 768) CLS representation
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        # [CLS] token representation
        sequence_output = outputs.last_hidden_state  # (B, T, 768)
        pooled_output = sequence_output[:, 0, :]     # (B, 768)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)      # (B, 3)

        result = {
            "logits": logits,
            "pooled_output": pooled_output,
            "hidden_states": outputs.hidden_states,   # (num_layers+1,) each (B, T, 768)
            "attentions": outputs.attentions,          # (num_layers,) each (B, 12, T, T)
        }

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 3,
        **kwargs,
    ) -> "PhoBERTTeacher":
        """Load a fine-tuned teacher from a local checkpoint."""
        model = cls(model_name=model_name, num_labels=num_labels, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle both raw state_dict and checkpoint dicts
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


def get_teacher_tokenizer(model_name: str = "vinai/phobert-base") -> AutoTokenizer:
    """Load PhoBERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[Teacher] Tokenizer loaded: {model_name} | Vocab size: {tokenizer.vocab_size}")
    return tokenizer
