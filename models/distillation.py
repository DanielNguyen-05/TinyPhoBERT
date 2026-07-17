"""
models/distillation.py

Multi-Level Knowledge Distillation Framework.

Implements the three-level distillation loss:

    L = L_CE + α·L_KD + β·L_hidden + γ·L_att

Where:
    L_CE     = Focal Loss hoặc Weighted Cross-Entropy (ground-truth labels)
    L_KD     = KL Divergence (soft logits from teacher, temperature-scaled)
    L_hidden = MSE + Cosine (hidden state alignment: student layer i ← teacher layer 2i)
    L_att    = MSE (attention map alignment)

Layer Mapping (default, 6 student → 12 teacher):
    Student [0,1,2,3,4,5] → Teacher [1,3,5,7,9,11]

Cải tiến so với version gốc:
    1. FocalLoss — giải quyết class imbalance (OFFENSIVE chỉ 7% data)
    2. class_weights — inverse-frequency weighting cho CrossEntropy
    3. label_smoothing — tránh overconfidence
    4. Cosine similarity loss cho hidden states — ổn định hơn MSE với scale khác nhau

Reference:
    DistilBERT (Sanh et al., 2019)
    TinyBERT (Jiao et al., 2020)
    PKD-BERT (Sun et al., 2019)
    Focal Loss (Lin et al., 2017)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.teacher import PhoBERTTeacher
from models.student import TinyPhoBERT


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss cho class imbalance (Lin et al., 2017).

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Ý tưởng:
        - Giảm contribution của easy examples (CLEAN, model dự đoán đúng dễ)
        - Tập trung học hard examples (OFFENSIVE, HATE)

    Args:
        gamma: Focusing parameter. γ=0 → CE thông thường. γ=2 (khuyến nghị).
        weight: Class weights tensor (α). Shape (num_classes,).
        label_smoothing: Label smoothing value [0, 1).
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, C) raw logits.
            target: (B,) integer class labels.
        """
        # Move weight to same device as input
        weight = self.weight
        if weight is not None:
            weight = weight.to(input.device)

        # Standard CE with label smoothing, per-sample
        ce_loss = F.cross_entropy(
            input, target,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # p_t = exp(-CE) (probability of the correct class)
        pt = torch.exp(-F.cross_entropy(input, target, reduction="none"))

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ── Multi-Level Distillation Loss ─────────────────────────────────────────────

class MultiLevelDistillationLoss(nn.Module):
    """
    Computes the multi-level distillation loss.

    L = L_CE + α·L_KD + β·L_hidden + γ·L_att

    Args:
        alpha: Weight for logit KD loss (KL divergence).
        beta: Weight for hidden state KD loss.
        gamma: Weight for attention KD loss (MSE).
        temperature: Softmax temperature for logit distillation.
        layer_mapping: Dict mapping student layer index → teacher layer index.
        use_logit_kd: Enable logit distillation.
        use_hidden_kd: Enable hidden state distillation.
        use_attention_kd: Enable attention map distillation.
        class_weights: Tensor of shape (num_classes,) for weighted loss.
                       Nếu None, tự động tính từ inverse frequency.
        use_focal_loss: Nếu True, dùng Focal Loss thay Weighted CE.
        focal_gamma: Gamma parameter cho Focal Loss (mặc định 2.0).
        label_smoothing: Label smoothing [0, 1) cho task loss.
        hidden_loss_type: 'mse' | 'cosine' | 'both' cho hidden state loss.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.1,
        temperature: float = 4.0,
        layer_mapping: Optional[Dict[int, int]] = None,
        use_logit_kd: bool = True,
        use_hidden_kd: bool = True,
        use_attention_kd: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        hidden_loss_type: str = "cosine",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.use_logit_kd = use_logit_kd
        self.use_hidden_kd = use_hidden_kd
        self.use_attention_kd = use_attention_kd
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.hidden_loss_type = hidden_loss_type

        # Default: student layer i maps to teacher layer (2*i + 1)
        if layer_mapping is None:
            self.layer_mapping = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11}
        else:
            self.layer_mapping = {int(k): int(v) for k, v in layer_mapping.items()}

        # ── Task Loss ──────────────────────────────────────────────────────────
        if use_focal_loss:
            # Focal Loss giải quyết class imbalance tốt hơn Weighted CE
            self.task_loss = FocalLoss(
                gamma=focal_gamma,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        else:
            # Weighted CrossEntropy với label smoothing
            self.task_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )

    # ── Helper: move class weights to device ──────────────────────────────────
    def _sync_weights_to_device(self, device: torch.device) -> None:
        """Đảm bảo class_weights và task_loss weights ở đúng device."""
        if self.class_weights is not None and self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
            # Cập nhật weight trong loss function
            if hasattr(self.task_loss, "weight") and self.task_loss.weight is not None:
                self.task_loss.weight = self.class_weights
            elif hasattr(self.task_loss, "_weight"):
                self.task_loss._weight = self.class_weights

    # ── Level 1: Logit KD ─────────────────────────────────────────────────────
    def logit_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Level 1: Logit Distillation via KL Divergence.

        L_KD = T² · KL(softmax(z_t/T) || log_softmax(z_s/T))

        Temperature T² scaling đảm bảo loss magnitude ổn định
        khi thay đổi temperature (Hinton et al., 2015).
        """
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return loss * (T ** 2)

    # ── Level 2: Hidden State KD ──────────────────────────────────────────────
    def hidden_state_loss(
        self,
        student_hidden: Tuple[torch.Tensor, ...],
        teacher_hidden: Tuple[torch.Tensor, ...],
        student_projected: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Level 2: Hidden State Distillation.

        Hỗ trợ 3 chế độ:
          'mse'    → MSE(mean-pooled student, mean-pooled teacher)
          'cosine' → 1 - cosine_similarity(student, teacher)  [ổn định hơn với scale khác nhau]
          'both'   → 0.5*MSE + 0.5*cosine

        L_hidden = (1/N) Σ_i loss(project(H_student_i), H_teacher_map[i])
        """
        total_loss = torch.tensor(0.0, device=attention_mask.device)
        count = 0

        mask = attention_mask.unsqueeze(-1).float()

        for s_idx, t_idx in self.layer_mapping.items():
            s_layer = s_idx + 1
            t_layer = t_idx + 1

            if s_layer >= len(student_projected) or t_layer >= len(teacher_hidden):
                continue

            # projected student: (B, T, 768)
            s_repr = student_projected[s_layer]
            # teacher: (B, T, 768)
            t_repr = teacher_hidden[t_layer].detach()

            # Mean-pool over real tokens only
            denom = mask.sum(dim=1).clamp(min=1e-9)
            s_mean = (s_repr * mask).sum(dim=1) / denom   # (B, 768)
            t_mean = (t_repr * mask).sum(dim=1) / denom   # (B, 768)

            if self.hidden_loss_type == "mse":
                layer_loss = F.mse_loss(s_mean, t_mean)
            elif self.hidden_loss_type == "cosine":
                # cosine_similarity returns (B,), we want mean over batch
                cos_sim = F.cosine_similarity(s_mean, t_mean, dim=-1)  # (B,)
                layer_loss = (1.0 - cos_sim).mean()
            else:  # "both"
                mse = F.mse_loss(s_mean, t_mean)
                cos_sim = F.cosine_similarity(s_mean, t_mean, dim=-1).mean()
                layer_loss = 0.5 * mse + 0.5 * (1.0 - cos_sim)

            total_loss = total_loss + layer_loss
            count += 1

        return total_loss / max(count, 1)

    # ── Level 3: Attention KD ─────────────────────────────────────────────────
    def attention_loss(
        self,
        student_attentions: Optional[Tuple[torch.Tensor, ...]],
        teacher_attentions: Optional[Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """
        Level 3: Attention Map Distillation via MSE.

        L_att = (1/N) Σ_i MSE(A_student_i, A_teacher_map[i])
        """
        if not student_attentions or not teacher_attentions:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0, device=student_attentions[0].device)
        count = 0

        for s_idx, t_idx in self.layer_mapping.items():
            if s_idx >= len(student_attentions) or t_idx >= len(teacher_attentions):
                continue

            s_att = student_attentions[s_idx].mean(dim=1)        # avg heads: (B,T,T)
            t_att = teacher_attentions[t_idx].detach().mean(dim=1)

            s_att = s_att.clamp(min=1e-8)
            t_att = t_att.clamp(min=1e-8)

            total_loss = total_loss + F.mse_loss(s_att, t_att)
            count += 1

        return total_loss / max(count, 1)

    # ── Forward ───────────────────────────────────────────────────────────────
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

        L = L_task + α·L_KD + β·L_hidden + γ·L_att

        Returns:
            Dictionary with individual losses and total loss.
        """
        # Sync class weights device
        self._sync_weights_to_device(student_logits.device)

        losses = {}

        # --- Task Loss (Focal or Weighted CE) ---
        l_ce = self.task_loss(student_logits, labels)
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


# ── DistillationTrainer ───────────────────────────────────────────────────────

class DistillationTrainer:
    """
    High-level trainer for multi-level knowledge distillation.

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
        """Single distillation step."""
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


# ── Factory ───────────────────────────────────────────────────────────────────

def build_distillation_loss_from_config(
    config: dict,
    class_weights: Optional[torch.Tensor] = None,
) -> MultiLevelDistillationLoss:
    """
    Build MultiLevelDistillationLoss from a config dictionary.

    Args:
        config: Full config dict (hoặc chỉ distillation sub-dict).
        class_weights: Optional tensor of shape (num_classes,).
    """
    distill_cfg = config.get("distillation", config)
    student_cfg = config.get("student", {})

    layer_mapping_raw = student_cfg.get("model", {}).get("layer_mapping", None)
    layer_mapping = None
    if layer_mapping_raw:
        layer_mapping = {int(k): int(v) for k, v in layer_mapping_raw.items()}

    return MultiLevelDistillationLoss(
        alpha=distill_cfg.get("alpha", 0.5),
        beta=distill_cfg.get("beta", 0.3),
        gamma=distill_cfg.get("gamma", 0.1),
        temperature=distill_cfg.get("temperature", 4.0),
        layer_mapping=layer_mapping,
        use_logit_kd=distill_cfg.get("use_logit_kd", True),
        use_hidden_kd=distill_cfg.get("use_hidden_kd", True),
        use_attention_kd=distill_cfg.get("use_attention_kd", True),
        class_weights=class_weights,
        use_focal_loss=distill_cfg.get("use_focal_loss", True),
        focal_gamma=distill_cfg.get("focal_gamma", 2.0),
        label_smoothing=distill_cfg.get("label_smoothing", 0.1),
        hidden_loss_type=distill_cfg.get("hidden_loss_type", "cosine"),
    )
