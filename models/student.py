"""
models/student.py

TinyPhoBERT — Compact Student Model cho Multi-Teacher Distillation.
6 layers / 384 hidden / 6 heads — nhỏ hơn nhiều so với các expert teacher.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel


class MultiScaleHateSpeechHead(nn.Module):
    """Learned upper-layer mix with global pooling and local n-gram features."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        num_mixed_layers: int = 4,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5),
        cnn_channels: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_mixed_layers < 1:
            raise ValueError("num_mixed_layers must be positive")
        if not cnn_kernel_sizes or any(k < 1 or k % 2 == 0 for k in cnn_kernel_sizes):
            raise ValueError("cnn_kernel_sizes must contain positive odd integers")
        self.num_mixed_layers = num_mixed_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_mixed_layers))
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, cnn_channels, kernel_size=k, padding=k // 2)
            for k in cnn_kernel_sizes
        ])
        combined_size = 2 * hidden_size + len(cnn_kernel_sizes) * cnn_channels
        self.projection = nn.Linear(combined_size, hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        available = min(self.num_mixed_layers, len(hidden_states))
        weights = torch.softmax(self.layer_weights[-available:], dim=0)
        sequence = sum(
            weight * hidden
            for weight, hidden in zip(weights, hidden_states[-available:])
        )
        mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
        cls_pool = sequence[:, 0]
        mean_pool = (sequence * mask).sum(1) / mask.sum(1).clamp(min=1.0)

        # Padding hidden vectors are non-zero; clear them before convolution so
        # they cannot contaminate boundary n-grams.
        conv_input = (sequence * mask).transpose(1, 2)
        padding_mask = attention_mask.unsqueeze(1).bool()
        conv_pools = []
        for conv in self.convs:
            features = F.gelu(conv(conv_input))
            features = features.masked_fill(
                ~padding_mask, torch.finfo(features.dtype).min
            )
            conv_pools.append(features.max(dim=2).values)

        candidate = F.gelu(self.projection(
            torch.cat([cls_pool, mean_pool, *conv_pools], dim=-1)
        ))
        gate = torch.sigmoid(self.gate(torch.cat([cls_pool, candidate], dim=-1)))
        pooled = self.dropout(self.norm(cls_pool + gate * candidate))
        return self.classifier(pooled), pooled


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
        teacher_hidden_size: int = 1024,
        layer_norm_eps: float = 1e-5,
        type_vocab_size: int = 1,
        classification_head: str = "linear",
        num_mixed_layers: int = 4,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5),
        cnn_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.teacher_hidden_size = teacher_hidden_size
        self.classification_head = classification_head

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
        self.config._attn_implementation = "eager"
        self.backbone = RobertaModel(self.config, add_pooling_layer=False)
        self.dropout = nn.Dropout(classifier_dropout)
        if classification_head == "linear":
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.multiscale_head = None
        elif classification_head == "multiscale":
            self.classifier = None
            self.multiscale_head = MultiScaleHateSpeechHead(
                hidden_size, num_labels, num_mixed_layers,
                tuple(cnn_kernel_sizes), cnn_channels, classifier_dropout,
            )
        else:
            raise ValueError("classification_head must be 'linear' or 'multiscale'")
        self.hidden_projection = nn.Linear(
            hidden_size, teacher_hidden_size, bias=False
        )

        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.hidden_projection.weight, std=0.02)

    def init_from_teacher(
        self,
        teacher_model,
        layer_mapping: Optional[dict] = None,
    ) -> None:
        """Initialize the compact backbone by slicing mapped teacher layers."""
        teacher_backbone = getattr(teacher_model, "backbone", teacher_model)
        teacher_layers = teacher_backbone.encoder.layer
        student_layers = self.backbone.encoder.layer
        teacher_depth = len(teacher_layers)
        teacher_width = teacher_backbone.config.hidden_size
        student_width = self.hidden_size
        if teacher_width < student_width:
            raise ValueError(
                f"Teacher hidden size {teacher_width} is smaller than student "
                f"hidden size {student_width}; weight slicing is impossible."
            )
        if teacher_backbone.config.vocab_size != self.config.vocab_size:
            raise ValueError(
                "Teacher/student vocabularies differ; use logit-only "
                "distillation instead of weight slicing."
            )
        if layer_mapping is None:
            # End-aligned mapping includes the teacher's task-specific top layer.
            layer_mapping = {
                i: round((i + 1) * teacher_depth / self.num_layers) - 1
                for i in range(self.num_layers)
            }
        layer_mapping = {int(k): int(v) for k, v in layer_mapping.items()}

        with torch.no_grad():
            teacher_embeddings = teacher_backbone.embeddings
            student_embeddings = self.backbone.embeddings
            student_embeddings.word_embeddings.weight.copy_(
                teacher_embeddings.word_embeddings.weight[:, :student_width]
            )
            student_embeddings.position_embeddings.weight.copy_(
                teacher_embeddings.position_embeddings.weight[:, :student_width]
            )
            student_embeddings.token_type_embeddings.weight.copy_(
                teacher_embeddings.token_type_embeddings.weight[
                    : student_embeddings.token_type_embeddings.weight.shape[0],
                    :student_width,
                ]
            )
            student_embeddings.LayerNorm.weight.copy_(
                teacher_embeddings.LayerNorm.weight[:student_width]
            )
            student_embeddings.LayerNorm.bias.copy_(
                teacher_embeddings.LayerNorm.bias[:student_width]
            )

            student_intermediate = self.config.intermediate_size
            for student_idx, teacher_idx in layer_mapping.items():
                if student_idx >= len(student_layers) or teacher_idx >= len(teacher_layers):
                    raise IndexError(
                        f"Invalid layer mapping {student_idx}->{teacher_idx}"
                    )
                student_layer = student_layers[student_idx]
                teacher_layer = teacher_layers[teacher_idx]
                for name in ("query", "key", "value"):
                    source = getattr(teacher_layer.attention.self, name)
                    target = getattr(student_layer.attention.self, name)
                    target.weight.copy_(
                        source.weight[:student_width, :student_width]
                    )
                    target.bias.copy_(source.bias[:student_width])

                student_layer.attention.output.dense.weight.copy_(
                    teacher_layer.attention.output.dense.weight[
                        :student_width, :student_width
                    ]
                )
                student_layer.attention.output.dense.bias.copy_(
                    teacher_layer.attention.output.dense.bias[:student_width]
                )
                student_layer.attention.output.LayerNorm.weight.copy_(
                    teacher_layer.attention.output.LayerNorm.weight[:student_width]
                )
                student_layer.attention.output.LayerNorm.bias.copy_(
                    teacher_layer.attention.output.LayerNorm.bias[:student_width]
                )
                student_layer.intermediate.dense.weight.copy_(
                    teacher_layer.intermediate.dense.weight[
                        :student_intermediate, :student_width
                    ]
                )
                student_layer.intermediate.dense.bias.copy_(
                    teacher_layer.intermediate.dense.bias[:student_intermediate]
                )
                student_layer.output.dense.weight.copy_(
                    teacher_layer.output.dense.weight[
                        :student_width, :student_intermediate
                    ]
                )
                student_layer.output.dense.bias.copy_(
                    teacher_layer.output.dense.bias[:student_width]
                )
                student_layer.output.LayerNorm.weight.copy_(
                    teacher_layer.output.LayerNorm.weight[:student_width]
                )
                student_layer.output.LayerNorm.bias.copy_(
                    teacher_layer.output.LayerNorm.bias[:student_width]
                )
        print(f"[InitFromTeacher] Layer mapping: {layer_mapping}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_distill_outputs: bool = False,
        return_attentions: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if return_attentions is None:
            return_attentions = return_distill_outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(
                return_distill_outputs or self.multiscale_head is not None
            ),
            output_attentions=return_attentions,
        )
        if self.multiscale_head is not None:
            logits, pooled_output = self.multiscale_head(
                outputs.hidden_states, attention_mask
            )
        else:
            pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
            logits = self.classifier(pooled_output)

        result = {"logits": logits, "pooled_output": pooled_output}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        if return_distill_outputs:
            result["hidden_states"] = outputs.hidden_states
            result["attentions"] = outputs.attentions
            result["projected_hidden"] = tuple(
                self.hidden_projection(hidden)
                for hidden in outputs.hidden_states
            )
        return result

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def model_size_mb(self) -> float:
        size = sum(p.numel() * p.element_size() for p in self.parameters())
        size += sum(b.numel() * b.element_size() for b in self.buffers())
        return size / (1024 ** 2)

    def print_summary(self) -> None:
        print(
            f"TinyPhoBERT: {self.num_layers} layers, {self.hidden_size} hidden, "
            f"head={self.classification_head}, "
            f"params={self.count_parameters(False):,}, "
            f"size={self.model_size_mb():.1f} MB"
        )


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
        teacher_hidden_size=model_cfg.get("teacher_hidden_size", 1024),
        layer_norm_eps=model_cfg.get("layer_norm_eps", 1e-5),
        classification_head=model_cfg.get("classification_head", "linear"),
        num_mixed_layers=model_cfg.get("num_mixed_layers", 4),
        cnn_kernel_sizes=tuple(model_cfg.get("cnn_kernel_sizes", [1, 3, 5])),
        cnn_channels=model_cfg.get("cnn_channels", 128),
    )
