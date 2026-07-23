"""
models/student.py

TinyPhoBERT — Compact Student Model cho Multi-Teacher Distillation.
6 layers / 384 hidden / 6 heads — nhỏ hơn nhiều so với các expert teacher.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel


class TinyPhoBERT(nn.Module):
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
        layer_norm_eps: float = 1e-5,
        type_vocab_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers

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
        )
        self.backbone = RobertaModel(self.config, add_pooling_layer=False)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_student_from_config(config: dict) -> TinyPhoBERT:
    """
    Build TinyPhoBERT từ config dict — giữ lại để tương thích với
    models/__init__.py (import build_student_from_config) từ code cũ.
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
        layer_norm_eps=model_cfg.get("layer_norm_eps", 1e-5),
    )