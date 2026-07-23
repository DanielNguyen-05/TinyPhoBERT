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

from models.student import MultiScaleHateSpeechHead


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
        classification_head: str = "linear",
        num_mixed_layers: int = 4,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5),
        cnn_channels: int = 128,
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
        self.classification_head = classification_head
        self.output_attentions = output_attentions

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
        # Classification uses hidden states directly; the pretrained RoBERTa
        # pooler is unused and should not appear as an unoptimized parameter.
        if getattr(self.backbone, "pooler", None) is not None:
            for parameter in self.backbone.pooler.parameters():
                parameter.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        if classification_head == "linear":
            self.classifier = nn.Linear(self.config.hidden_size, num_labels)
            self.multiscale_head = None
        elif classification_head == "multiscale":
            self.classifier = None
            self.multiscale_head = MultiScaleHateSpeechHead(
                hidden_size=self.config.hidden_size,
                num_labels=num_labels,
                num_mixed_layers=num_mixed_layers,
                cnn_kernel_sizes=tuple(cnn_kernel_sizes),
                cnn_channels=cnn_channels,
                dropout=dropout,
            )
        else:
            raise ValueError("classification_head must be 'linear' or 'multiscale'")

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
            output_attentions=self.output_attentions,
        )

        if self.multiscale_head is not None:
            logits, pooled_output = self.multiscale_head(
                outputs.hidden_states, attention_mask
            )
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(self.dropout(pooled_output))

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
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False,)
        # New checkpoints carry the architecture settings. Reconstruct the
        # exact head automatically so distillation never falls back to a random
        # linear classifier when the teacher was trained with a multi-scale head.
        saved_config = state_dict.get("config", {}) if isinstance(state_dict, dict) else {}
        model_cfg = saved_config.get("model", {})
        training_cfg = saved_config.get("training", {})
        inferred = {
            "dropout": model_cfg.get("dropout", 0.1),
            "classification_head": model_cfg.get("classification_head", "linear"),
            "num_mixed_layers": model_cfg.get("num_mixed_layers", 4),
            "cnn_kernel_sizes": tuple(model_cfg.get("cnn_kernel_sizes", [1, 3, 5])),
            "cnn_channels": model_cfg.get("cnn_channels", 128),
            "use_focal_loss": training_cfg.get("use_focal_loss", True),
            "focal_gamma": training_cfg.get("focal_gamma", 2.0),
            "label_smoothing": training_cfg.get("label_smoothing", 0.1),
            "use_supcon": training_cfg.get("use_supcon", False),
            "supcon_proj_dim": training_cfg.get("supcon_proj_dim", 256),
            "supcon_weight": training_cfg.get("supcon_weight", 0.1),
            "supcon_temperature": training_cfg.get("supcon_temperature", 0.07),
        }
        inferred.update(kwargs)
        saved_model_name = model_cfg.get("name", model_name)
        model = cls(model_name=saved_model_name, num_labels=num_labels, **inferred)
        # Handle both raw state_dict and checkpoint dicts
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        incompatible = model.load_state_dict(state_dict, strict=False)
        critical_missing = [
            key for key in incompatible.missing_keys
            if key.startswith(("backbone.", "classifier.", "multiscale_head."))
        ]
        critical_unexpected = [
            key for key in incompatible.unexpected_keys
            if key.startswith(("backbone.", "classifier.", "multiscale_head."))
        ]
        if critical_missing or critical_unexpected:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Missing={critical_missing[:10]}, "
                f"unexpected={critical_unexpected[:10]}"
            )
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
