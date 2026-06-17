"""
models/distillation.py

Multi-Level Knowledge Distillation Framework.

Implements the three-level distillation loss:

    L = L_CE + α·L_KD + β·L_hidden + γ·L_att

Where:
    L_CE     = Cross-Entropy (ground-truth labels)
    L_KD     = KL Divergence (soft logits from teacher, temperature-scaled)
    L_hidden = MSE (hidden state alignment: student layer i ← teacher layer 2i)
    L_att    = MSE (attention map alignment)

Layer Mapping (default, 6 student → 12 teacher):
    Student [0,1,2,3,4,5] → Teacher [1,3,5,7,9,11]

Reference:
    DistilBERT (Sanh et al., 2019)
    TinyBERT (Jiao et al., 2020)
    PKD-BERT (Sun et al., 2019)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.teacher import PhoBERTTeacher
from models.student import TinyPhoBERT


class MultiLevelDistillationLoss(nn.Module):
    """
    Computes the multi-level distillation loss.

    Args:
        alpha: Weight for logit KD loss (KL divergence).
        beta: Weight for hidden state KD loss (MSE).
        gamma: Weight for attention KD loss (MSE).
        temperature: Softmax temperature for logit distillation.
        layer_mapping: Dict mapping student layer index → teacher layer index.
        use_logit_kd: Enable logit distillation.
        use_hidden_kd: Enable hidden state distillation.
        use_attention_kd: Enable attention map distillation.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 0.1,
        temperature: float = 4.0,
        layer_mapping: Optional[Dict[int, int]] = None,
        use_logit_kd: bool = True,
        use_hidden_kd: bool = True,
        use_attention_kd: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.use_logit_kd = use_logit_kd
        self.use_hidden_kd = use_hidden_kd
        self.use_attention_kd = use_attention_kd

        # Default: student layer i maps to teacher layer (2*i + 1)
        if layer_mapping is None:
            self.layer_mapping = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11}
        else:
            self.layer_mapping = {int(k): int(v) for k, v in layer_mapping.items()}

        self.ce_loss = nn.CrossEntropyLoss()

    def logit_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Level 1: Logit Distillation via KL Divergence.

        L_KD = T² · KL(softmax(z_t/T) || log_softmax(z_s/T))

        The temperature T² scaling ensures the loss magnitude is
        comparable across different temperatures (Hinton et al., 2015).

        Args:
            student_logits: (B, num_labels) raw student logits.
            teacher_logits: (B, num_labels) raw teacher logits.

        Returns:
            Scalar KL divergence loss.
        """
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return loss * (T ** 2)

    def hidden_state_loss(
        self,
        student_hidden: Tuple[torch.Tensor, ...],
        teacher_hidden: Tuple[torch.Tensor, ...],
        student_projected: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Level 2: Hidden State Distillation via MSE.

        L_hidden = (1/N) Σ_i MSE(project(H_student_i), H_teacher_map[i])

        Uses mean-pooled representation over non-padding tokens to focus
        the loss on content tokens rather than [PAD].

        Args:
            student_hidden: Tuple of (B, T, 384) student hidden states (all layers).
            teacher_hidden: Tuple of (B, T, 768) teacher hidden states (all layers).
            student_projected: Tuple of (B, T, 768) projected student states.
            attention_mask: (B, T) attention mask (1=real token, 0=padding).

        Returns:
            Scalar MSE loss averaged over layers.
        """
        total_loss = torch.tensor(0.0, device=attention_mask.device)
        count = 0

        # Expand mask for broadcasting: (B, T, 1)
        mask = attention_mask.unsqueeze(-1).float()

        for s_idx, t_idx in self.layer_mapping.items():
            # student_hidden has +1 because index 0 is embedding layer
            s_layer = s_idx + 1  # student index in hidden_states tuple
            t_layer = t_idx + 1  # teacher index in hidden_states tuple

            if s_layer >= len(student_projected) or t_layer >= len(teacher_hidden):
                continue

            # projected student: (B, T, 768)
            s_repr = student_projected[s_layer]
            # teacher: (B, T, 768)
            t_repr = teacher_hidden[t_layer].detach()

            # Mean-pool over sequence (only real tokens)
            s_mean = (s_repr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            t_mean = (t_repr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            total_loss = total_loss + F.mse_loss(s_mean, t_mean)
            count += 1

        return total_loss / max(count, 1)

    def attention_loss(
        self,
        student_attentions: Optional[Tuple[torch.Tensor, ...]],
        teacher_attentions: Optional[Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """
        Level 3: Attention Map Distillation via MSE.

        L_att = (1/N) Σ_i MSE(A_student_i, A_teacher_map[i])

        Attention maps are first averaged across heads, then MSE is
        computed. Teacher values are detached from the computation graph.

        Args:
            student_attentions: Tuple of (B, 6, T, T) per student layer.
            teacher_attentions: Tuple of (B, 12, T, T) per teacher layer.

        Returns:
            Scalar MSE loss averaged over layers.
        """
        # Guard: if attentions are unavailable (e.g., SDPA mode), return zero
        if not student_attentions or not teacher_attentions:
            dummy_device = next(iter(self.parameters())).device if list(self.parameters()) else torch.device("cpu")
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0, device=student_attentions[0].device)
        count = 0

        for s_idx, t_idx in self.layer_mapping.items():
            if s_idx >= len(student_attentions) or t_idx >= len(teacher_attentions):
                continue

            # Average over heads: (B, T, T)
            s_att = student_attentions[s_idx].mean(dim=1)
            t_att = teacher_attentions[t_idx].detach().mean(dim=1)

            # Clamp attention values to avoid log(0) issues
            s_att = s_att.clamp(min=1e-8)
            t_att = t_att.clamp(min=1e-8)

            total_loss = total_loss + F.mse_loss(s_att, t_att)
            count += 1

        return total_loss / max(count, 1)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        teacher_hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        student_projected: Optional[Tuple[torch.Tensor, ...]] = None,
        student_attentions: Optional[Tuple[torch.Tensor, ...]] = None,
        teacher_attentions: Optional[Tuple[torch.Tensor, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total multi-level distillation loss.

        L = L_CE + α·L_KD + β·L_hidden + γ·L_att

        Returns:
            Dictionary with individual losses and total loss.
        """
        losses = {}

        # --- Task Loss (always enabled) ---
        l_ce = self.ce_loss(student_logits, labels)
        losses["loss_ce"] = l_ce

        # --- Level 1: Logit Distillation ---
        l_kd = torch.tensor(0.0, device=student_logits.device)
        if self.use_logit_kd and self.alpha > 0:
            l_kd = self.logit_kd_loss(student_logits, teacher_logits)
        losses["loss_kd"] = l_kd

        # --- Level 2: Hidden State Distillation ---
        l_hidden = torch.tensor(0.0, device=student_logits.device)
        if (
            self.use_hidden_kd
            and self.beta > 0
            and student_hidden is not None
            and teacher_hidden is not None
            and student_projected is not None
            and attention_mask is not None
        ):
            l_hidden = self.hidden_state_loss(
                student_hidden, teacher_hidden, student_projected, attention_mask
            )
        losses["loss_hidden"] = l_hidden

        # --- Level 3: Attention Distillation ---
        l_att = torch.tensor(0.0, device=student_logits.device)
        if (
            self.use_attention_kd
            and self.gamma > 0
            and student_attentions is not None
            and teacher_attentions is not None
        ):
            l_att = self.attention_loss(student_attentions, teacher_attentions)
        losses["loss_att"] = l_att

        # --- Total Loss ---
        total = l_ce + self.alpha * l_kd + self.beta * l_hidden + self.gamma * l_att
        losses["loss"] = total

        return losses


class DistillationTrainer:
    """
    High-level trainer for multi-level knowledge distillation.

    Manages the teacher (frozen) and student models, distillation loss,
    optimizer, and training loop.

    Args:
        teacher: Frozen PhoBERTTeacher model.
        student: TinyPhoBERT student model.
        distill_loss: MultiLevelDistillationLoss instance.
        device: Torch device.
    """

    def __init__(
        self,
        teacher: PhoBERTTeacher,
        student: TinyPhoBERT,
        distill_loss: MultiLevelDistillationLoss,
        device: torch.device,
    ) -> None:
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.distill_loss = distill_loss.to(device)
        self.device = device

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distill_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Single distillation step.

        Args:
            batch: Dict with input_ids, attention_mask, labels (all on device).

        Returns:
            Dict with all individual losses and the total loss.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Student forward (with grad)
        need_distill = (
            self.distill_loss.use_hidden_kd or self.distill_loss.use_attention_kd
        )
        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_distill_outputs=need_distill,
        )

        # Compute distillation loss
        losses = self.distill_loss(
            student_logits=student_out["logits"],
            teacher_logits=teacher_out["logits"],
            labels=labels,
            student_hidden=student_out.get("hidden_states"),
            teacher_hidden=teacher_out.get("hidden_states"),
            student_projected=student_out.get("projected_hidden"),
            student_attentions=student_out.get("attentions"),
            teacher_attentions=teacher_out.get("attentions"),
            attention_mask=attention_mask,
        )

        return losses


def build_distillation_loss_from_config(config: dict) -> MultiLevelDistillationLoss:
    """
    Build MultiLevelDistillationLoss from a config dictionary.

    Args:
        config: Distillation config dict.

    Returns:
        Configured MultiLevelDistillationLoss instance.
    """
    distill_cfg = config.get("distillation", config)
    student_cfg = config.get("student", {})

    # Layer mapping from student config
    layer_mapping_raw = student_cfg.get("model", {}).get("layer_mapping", None)
    layer_mapping = None
    if layer_mapping_raw:
        layer_mapping = {int(k): int(v) for k, v in layer_mapping_raw.items()}

    return MultiLevelDistillationLoss(
        alpha=distill_cfg.get("alpha", 0.5),
        beta=distill_cfg.get("beta", 0.1),
        gamma=distill_cfg.get("gamma", 0.1),
        temperature=distill_cfg.get("temperature", 4.0),
        layer_mapping=layer_mapping,
        use_logit_kd=distill_cfg.get("use_logit_kd", True),
        use_hidden_kd=distill_cfg.get("use_hidden_kd", True),
        use_attention_kd=distill_cfg.get("use_attention_kd", True),
    )
