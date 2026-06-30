"""
models/teacher.py

Teacher Model for Vietnamese Hate Speech Detection.

Hỗ trợ cả PhoBERT-base (12L/768H) và PhoBERT-large (24L/1024H).
Teacher chỉ cần load 1 lần và freeze → kích thước không ảnh hưởng
đến inference speed của student sau khi distill xong.
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
        model_name: str = "vinai/phobert-large",
        num_labels: int = 3,
        dropout: float = 0.1,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_supcon: bool = False,
        supcon_proj_dim: int = 256,
        supcon_weight: float = 0.3,
        supcon_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_supcon = use_supcon
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature

        # Load backbone — override dropout prob để khớp với classifier dropout.
        # Khi dùng SupCon, model dễ overfit hơn (embedding sắc nét, học thuộc
        # augmented samples) → cần regularize cả backbone, không chỉ head.
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=max(0.0, dropout - 0.1),
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # SupCon projection head (tùy chọn) — map pooled_output sang không
        # gian contrastive riêng, tách biệt với không gian dùng để classify.
        if self.use_supcon:
            from models.supcon_loss import SupConProjectionHead
            self.supcon_proj = SupConProjectionHead(
                input_dim=self.config.hidden_size,
                proj_dim=supcon_proj_dim,
            )
        else:
            self.supcon_proj = None

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

        pooled_output_dropped = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_dropped)      # (B, 3)

        result = {
            "logits": logits,
            "pooled_output": pooled_output,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

        # SupCon dùng pooled_output TRƯỚC dropout (embedding "sạch", ổn định
        # hơn cho việc đo similarity) — qua projection head nếu có.
        if self.use_supcon and self.supcon_proj is not None:
            result["supcon_features"] = self.supcon_proj(pooled_output)

        if labels is not None:
            result["loss"] = self._compute_loss(logits, labels, result.get("supcon_features"))

        return result

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        supcon_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Focal Loss / Weighted CE, cộng thêm SupCon loss nếu bật."""
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
            task_loss = (focal_w * ce).mean()
        else:
            task_loss = F.cross_entropy(
                logits, labels,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )

        if self.use_supcon and supcon_features is not None:
            from models.supcon_loss import SupConLoss
            if not hasattr(self, "_supcon_loss_fn"):
                self._supcon_loss_fn = SupConLoss(temperature=self.supcon_temperature)
            supcon_loss = self._supcon_loss_fn(supcon_features, labels)
            return task_loss + self.supcon_weight * supcon_loss

        return task_loss

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
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False,)
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


def get_teacher_tokenizer(model_name: str = "vinai/phobert-large") -> AutoTokenizer:
    """Load PhoBERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[Teacher] Tokenizer loaded: {model_name} | Vocab size: {tokenizer.vocab_size}")
    return tokenizer