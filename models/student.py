"""
models/student.py

TinyPhoBERT Student Model — Compact Vietnamese Hate Speech Classifier.

Architecture:
    6 Transformer Layers (vs 12 in PhoBERT)
    384 Hidden Size    (vs 768 in PhoBERT)
    6 Attention Heads  (vs 12 in PhoBERT)
    ~35-45M parameters

The student uses the same BPE tokenizer as PhoBERT (vinai/phobert-base),
ensuring identical tokenization between teacher and student.

A linear projection layer (384 → 768) is added to align student hidden
states with teacher hidden states during distillation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel


class TinyPhoBERT(nn.Module):
    """
    TinyPhoBERT: Compact student model for Vietnamese Hate Speech Detection.

    Args:
        vocab_size: Vocabulary size (matches PhoBERT: 64001).
        num_hidden_layers: Number of Transformer layers (default: 6).
        hidden_size: Hidden dimension size (default: 384).
        num_attention_heads: Number of attention heads (default: 6).
        intermediate_size: FFN intermediate size (default: 1536 = 4 × 384).
        max_position_embeddings: Max sequence length (default: 258, same as PhoBERT).
        num_labels: Number of output classes (default: 3).
        hidden_dropout_prob: Dropout on hidden states.
        attention_probs_dropout_prob: Dropout on attention weights.
        classifier_dropout: Dropout before classifier head.
        teacher_hidden_size: Teacher's hidden size for projection (default: 768).
    """

    def __init__(
        self,
        vocab_size: int = 64001,
        num_hidden_layers: int = 6,
        hidden_size: int = 384,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        max_position_embeddings: int = 258,
        num_labels: int = 3,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        teacher_hidden_size: int = 768,
        layer_norm_eps: float = 1e-5,
        type_vocab_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.teacher_hidden_size = teacher_hidden_size

        # Build RoBERTa config (PhoBERT uses RoBERTa architecture)
        # NOTE: attn_implementation="eager" is required for output_attentions=True
        # in newer transformers versions that default to SDPA attention.
        self.config = RobertaConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            type_vocab_size=type_vocab_size,
            output_hidden_states=True,
            output_attentions=True,
        )

        # Main transformer backbone
        # Set _attn_implementation to "eager" so that output_attentions=True
        # works correctly in newer transformers that default to SDPA.
        self.config._attn_implementation = "eager"
        self.backbone = RobertaModel(self.config, add_pooling_layer=False)


        # Classifier head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Projection layer: align student hidden states (384) → teacher (768)
        # Used ONLY during distillation, not for inference
        self.hidden_projection = nn.Linear(hidden_size, teacher_hidden_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize classifier and projection with small normal weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.hidden_projection.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_distill_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (B, T) token IDs.
            attention_mask: (B, T) attention mask.
            labels: (B,) integer class labels (optional).
            return_distill_outputs: If True, return hidden states and attentions
                                    for knowledge distillation.

        Returns:
            Dictionary with:
                logits          : (B, num_labels)
                loss            : scalar CE loss (if labels provided)
                pooled_output   : (B, 384) CLS representation
                hidden_states   : tuple of (B, T, 384) per layer [if distill]
                projected_hidden: tuple of (B, T, 768) projected [if distill]
                attentions      : tuple of (B, 6, T, T) per layer [if distill]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_distill_outputs,
            output_attentions=return_distill_outputs,
        )

        sequence_output = outputs.last_hidden_state   # (B, T, 384)
        cls_output = sequence_output[:, 0, :]          # (B, 384)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)           # (B, 3)

        result = {
            "logits": logits,
            "pooled_output": cls_output,
        }

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        if return_distill_outputs:
            result["hidden_states"] = outputs.hidden_states   # (num_layers+1,)
            result["attentions"] = outputs.attentions          # (num_layers,)

            # Project all hidden states for alignment with teacher
            projected = tuple(
                self.hidden_projection(h) for h in outputs.hidden_states
            )
            result["projected_hidden"] = projected

        return result

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def model_size_mb(self) -> float:
        """Return approximate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def print_summary(self) -> None:
        """Print model architecture summary."""
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        size_mb = self.model_size_mb()
        print("=" * 55)
        print("TinyPhoBERT Model Summary")
        print("=" * 55)
        print(f"  Transformer Layers  : {self.num_layers}")
        print(f"  Hidden Size         : {self.hidden_size}")
        print(f"  Attention Heads     : {self.num_heads}")
        print(f"  Total Parameters    : {total:,}")
        print(f"  Trainable Params    : {trainable:,}")
        print(f"  Model Size          : {size_mb:.1f} MB")
        print("=" * 55)


def build_student_from_config(config: dict) -> TinyPhoBERT:
    """
    Build TinyPhoBERT from a config dictionary (loaded from YAML).

    Args:
        config: Dict with keys matching TinyPhoBERT.__init__ arguments.

    Returns:
        Initialized TinyPhoBERT model.
    """
    model_cfg = config.get("model", config)
    return TinyPhoBERT(
        vocab_size=model_cfg.get("vocab_size", 64001),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 6),
        hidden_size=model_cfg.get("hidden_size", 384),
        num_attention_heads=model_cfg.get("num_attention_heads", 6),
        intermediate_size=model_cfg.get("intermediate_size", 1536),
        max_position_embeddings=model_cfg.get("max_position_embeddings", 258),
        num_labels=model_cfg.get("num_labels", 3),
        hidden_dropout_prob=model_cfg.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=model_cfg.get("attention_probs_dropout_prob", 0.1),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        teacher_hidden_size=model_cfg.get("teacher_hidden_size", 768),
        layer_norm_eps=model_cfg.get("layer_norm_eps", 1e-5),
    )
