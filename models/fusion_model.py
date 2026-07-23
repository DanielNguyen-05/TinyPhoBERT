"""
models/fusion_model.py

LLM-Fused PhoBERT for Vietnamese Hate Speech Detection — v2.

Kiến trúc v2 (so với v1):
    ┌──────────────────────────────────────────────────────────┐
    │  Input Vietnamese comment                                │
    │         │                                                │
    │    ┌────┴────────────────────────┐                       │
    │    ▼                             ▼                       │
    │  PhoBERT-large (trainable)   Qwen2.5-0.5B (frozen, npy) │
    │         │                         │                      │
    │  Hybrid Pooling:           LLM Projection:               │
    │  CLS + mean + max          896 → 512 (GELU + LN)        │
    │  (3072-dim) → proj → 1024        │                      │
    │         │                         │                      │
    │         └──── Cross-Attention ────┘  [NEW v2]            │
    │                      │                                   │
    │               Fusion MLP:                                │
    │               Linear(1024→512)→LN→GELU→Dropout           │
    │               ├── Linear(512→3)  [classification]        │
    │               └── SupCon proj (256-dim) [contrastive]    │
    └──────────────────────────────────────────────────────────┘

Cải tiến v2 so với v1:
    1. Fix Focal Loss bug — tách label_smoothing khỏi focal weight
    2. Hybrid Pooling: CLS + mean_pool + max_pool → richer PhoBERT repr
    3. Cross-Attention Fusion: PhoBERT attend vào LLM thay vì naive concat
    4. SupCon Loss tích hợp vào Fusion Model (kéo OFFENSIVE/HATE embeddings xa CLEAN)
    5. Logit Adjustment: post-hoc calibration dựa trên class prior
    6. Residual connection trong FusionMLP

References:
    - Focal Loss: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    - SupCon: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    - Cross-attention fusion: LLMEmbed (ACL 2024)
    - Logit Adjustment: Menon et al. "Long-tail Learning via Logit Adjustment" (NeurIPS 2020)
    - Hybrid Pooling: SHIELD (WOAH/ACL 2024)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


# ══════════════════════════════════════════════════════════════════════════════
# Helper Modules
# ══════════════════════════════════════════════════════════════════════════════

class LLMProjectionHead(nn.Module):
    """
    Project LLM embedding (896-dim) sang fusion space (512-dim).
    v2: Thêm second Linear để học non-linear projection tốt hơn.
    """

    def __init__(self, llm_hidden_size: int = 896, proj_dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_hidden_size, proj_dim, bias=True),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim, bias=True),
            nn.LayerNorm(proj_dim),
        )
        # Xavier init cho cả 2 linear layers
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HybridPhoBERTPooling(nn.Module):
    """
    Hybrid Pooling: concat(CLS, mean_pool, max_pool) → project về phobert_dim.

    Tại sao hybrid > CLS-only?
        - CLS token học global representation nhưng với câu ngắn (~11 từ),
          nó có thể bị dominated bởi positional bias.
        - Mean-pool: bao gồm toàn bộ context, robust với câu ngắn.
        - Max-pool: capture peak signal từ các offensive keywords.
        - Kết hợp 3 views → richer representation cho minority class OFFENSIVE/HATE.

    Reference: SHIELD (WOAH/ACL 2024), SentenceBERT ablation studies.
    """

    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        # CLS + mean + max = 3 * hidden_size → project về hidden_size
        self.proj = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        last_hidden_state: torch.Tensor,   # (B, T, H)
        attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:                      # (B, H)
        # CLS token
        cls = last_hidden_state[:, 0, :]    # (B, H)

        # Mean pool over non-padding tokens
        mask_exp = attention_mask.unsqueeze(-1).float()   # (B, T, 1)
        sum_hidden = (last_hidden_state * mask_exp).sum(1)  # (B, H)
        count = mask_exp.sum(1).clamp(min=1e-9)             # (B, 1)
        mean_pool = sum_hidden / count                       # (B, H)

        # Max pool over non-padding tokens
        large_neg = -1e4
        max_pool = last_hidden_state.masked_fill(
            attention_mask.unsqueeze(-1) == 0, large_neg
        ).max(dim=1).values                                 # (B, H)

        # Concat + project
        concat = torch.cat([cls, mean_pool, max_pool], dim=-1)  # (B, 3H)
        out = self.norm(self.proj(concat))                       # (B, H)
        return self.dropout(out)


class CrossAttentionFusion(nn.Module):
    """
    Gated interaction fusion for two pooled representations.

    Tại sao cross-attention > naive concat?
        - Naive concat xử lý 2 representations bình đẳng, bất kể relevance.
        - A learned gate lets PhoBERT control how much LLM context enters
          each feature dimension.
        - Đặc biệt hữu ích cho OFFENSIVE: PhoBERT thấy từ thô tục nhưng không
          chắc context → LLM cung cấp broader semantic context về hate patterns.

    Kiến trúc:
        Context = project(LLM)
        Gate    = sigmoid(W[PhoBERT; Context; PhoBERT⊙Context])
        Output  = LayerNorm(PhoBERT + Gate⊙Context)

    Reference: LLMEmbed "Rethinking Lightweight LLM's Genuine Function
               in Text Classification" (ACL 2024).
    """

    def __init__(
        self,
        phobert_dim: int = 1024,
        llm_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Both branches are single pooled vectors. Multi-head attention over a
        # one-token key/value sequence is degenerate: softmax always returns
        # 1, so the query cannot affect the result. Use a learned interaction
        # gate instead; it is conditioned on PhoBERT, the LLM, and their
        # element-wise agreement.
        self.llm_to_context = nn.Linear(llm_dim, phobert_dim, bias=False)
        self.gate = nn.Linear(3 * phobert_dim, phobert_dim)
        self.norm = nn.LayerNorm(phobert_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.llm_to_context.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        phobert_repr: torch.Tensor,  # (B, phobert_dim)
        llm_proj: torch.Tensor,      # (B, llm_dim)
    ) -> torch.Tensor:               # (B, phobert_dim)
        llm_context = self.llm_to_context(llm_proj)
        interaction = phobert_repr * llm_context
        gate = torch.sigmoid(self.gate(torch.cat(
            [phobert_repr, llm_context, interaction], dim=-1
        )))
        out = self.norm(phobert_repr + self.dropout(gate * llm_context))
        return out


class FusionMLP(nn.Module):
    """
    MLP fusion head với residual connection.
    v2: Thêm residual skip để tránh gradient vanishing khi SupCon điều chỉnh
    embedding space.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Residual projection nếu input_dim != hidden_dim
        self.residual = (
            nn.Linear(input_dim, hidden_dim, bias=False)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, hidden) — hidden dùng cho SupCon."""
        residual = self.residual(x)
        h = self.act(self.norm1(self.fc1(x)))
        h = self.dropout(h + residual)      # residual connection
        logits = self.classifier(h)
        return logits, h


class SupConProjectionHead(nn.Module):
    """
    Projection head cho Supervised Contrastive Learning.
    Map fusion hidden → contrastive space (256-dim).
    Tách biệt khỏi classification space để không force classifier embedding
    phải satisfy 2 objectives cùng lúc.
    """

    def __init__(self, input_dim: int = 512, proj_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ══════════════════════════════════════════════════════════════════════════════

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
    num_classes: int = 3,
) -> torch.Tensor:
    """
    Correct Focal Loss implementation — tách label_smoothing khỏi focal weight.

    Bug trong v1: label_smoothing được áp dụng vào ce TRƯỚC khi tính pt,
    khiến focal_weight = (1-pt)^gamma không consistent với actual ce.

    Fix v2:
        1. Tính pt từ CE THUẦN (không có smoothing) → focal weight chính xác
        2. Áp dụng class weight vào ce_raw TRƯỚC focal weight
        3. Tính label_smoothing loss riêng và cộng lại (như penalty riêng)

    Reference: Lin et al. (ICCV 2017), equation (4)-(5).
    """
    # pt từ CE không có smoothing — cơ sở để tính focal weight
    with torch.no_grad():
        pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
        focal_w = (1.0 - pt) ** gamma   # (B,)

    # CE với class weight (không có smoothing) để kết hợp với focal_w
    ce_raw = F.cross_entropy(logits, labels, weight=weight, reduction="none")

    # Focal loss term
    focal_term = (focal_w * ce_raw).mean()

    # Label smoothing regularization: KL(smooth_dist || model_dist)
    # Thêm riêng như regularizer với weight nhỏ
    if label_smoothing > 0:
        n_cls = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.mean(dim=-1).mean() * label_smoothing
        return focal_term + smooth_loss

    return focal_term


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Inline implementation để tránh import cycle.
    """
    device = features.device
    batch_size = features.shape[0]

    if batch_size <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    features = F.normalize(features, p=2, dim=1)
    labels = labels.contiguous().view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(device)
    sim = torch.matmul(features, features.T) / temperature

    # Numerical stability
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - sim_max.detach()

    # Remove self-similarity
    eye = torch.eye(batch_size, device=device)
    logits_mask = 1.0 - eye
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    num_pos = mask.sum(dim=1)
    valid = num_pos > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / num_pos[valid]
    loss = -mean_log_prob_pos
    return loss.mean()


def apply_logit_adjustment(
    logits: torch.Tensor,
    log_prior: Optional[torch.Tensor],
    tau: float,
) -> torch.Tensor:
    """Remove class-prior bias from decision scores at inference time."""
    if log_prior is None or tau <= 0:
        return logits
    return logits - tau * log_prior.to(device=logits.device, dtype=logits.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════

class LLMFusedPhoBERT(nn.Module):
    """
    PhoBERT-large + Frozen LLM embedding fusion model — v2.

    Args:
        model_name: PhoBERT model name (default: vinai/phobert-large)
        num_labels: Number of classification labels (default: 3)
        llm_hidden_size: Hidden size của LLM extractor (Qwen2.5-0.5B: 896)
        llm_proj_dim: Projection dimension cho LLM embedding (default: 512)
        phobert_dropout: Dropout trong HybridPhoBERTPooling
        fusion_dropout: Dropout trong FusionMLP
        class_weights: Optional class weights cho Focal loss
        use_focal_loss: Dùng Focal Loss thay vì CE
        focal_gamma: γ cho Focal Loss (default: 3.0 để phạt CLEAN mạnh hơn)
        label_smoothing: Label smoothing regularization
        use_supcon: Bật Supervised Contrastive Learning (default: True)
        supcon_weight: Weight của SupCon loss trong total loss
        supcon_temperature: Temperature τ cho SupCon
        use_cross_attention: Dùng gated interaction fusion thay vì concat
        use_hybrid_pooling: Dùng Hybrid Pooling (CLS+mean+max) thay vì CLS-only
        logit_adjustment_tau: τ cho Logit Adjustment (0 = tắt)
        class_prior: Class prior probabilities (dùng với logit_adjustment_tau)
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-large",
        num_labels: int = 3,
        llm_hidden_size: int = 896,
        llm_proj_dim: int = 512,
        phobert_dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 3.0,
        label_smoothing: float = 0.05,
        use_supcon: bool = True,
        supcon_weight: float = 0.1,
        supcon_temperature: float = 0.07,
        use_cross_attention: bool = True,
        use_hybrid_pooling: bool = True,
        logit_adjustment_tau: float = 0.0,
        class_prior: Optional[List[float]] = None,
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
        self.use_cross_attention = use_cross_attention
        self.use_hybrid_pooling = use_hybrid_pooling
        self.logit_adjustment_tau = logit_adjustment_tau

        # Logit Adjustment: log(π_c) per class
        # Ref: Menon et al. "Long-tail Learning via Logit Adjustment" (NeurIPS 2020)
        if logit_adjustment_tau > 0 and class_prior is not None:
            prior_tensor = torch.tensor(class_prior, dtype=torch.float32)
            # log_prior: (num_labels,). Subtracting it at inference removes
            # the training prior and raises minority-class decision scores.
            self.register_buffer("log_prior", torch.log(prior_tensor.clamp(min=1e-9)))
        else:
            self.register_buffer("log_prior", None)

        # ── PhoBERT backbone ──────────────────────────────────────────────────
        config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.phobert_hidden_size = config.hidden_size  # 1024 for phobert-large
        self.num_layers = config.num_hidden_layers

        # ── PhoBERT Pooling ───────────────────────────────────────────────────
        if use_hybrid_pooling:
            self.phobert_pooling = HybridPhoBERTPooling(
                hidden_size=self.phobert_hidden_size,
                dropout=phobert_dropout,
            )
            # Output dim after pooling
            self.phobert_repr_dim = self.phobert_hidden_size  # projected back to H
        else:
            self.phobert_dropout = nn.Dropout(phobert_dropout)
            self.phobert_repr_dim = self.phobert_hidden_size

        # ── LLM Projection head ───────────────────────────────────────────────
        self.llm_proj = LLMProjectionHead(llm_hidden_size, llm_proj_dim)

        # ── Fusion: Cross-Attention or Concat ─────────────────────────────────
        if use_cross_attention:
            self.cross_attn_fusion = CrossAttentionFusion(
                phobert_dim=self.phobert_repr_dim,
                llm_dim=llm_proj_dim,
                num_heads=8,
                dropout=fusion_dropout,
            )
            fusion_input_dim = self.phobert_repr_dim   # Cross-attn output = phobert_dim
        else:
            self.cross_attn_fusion = None
            fusion_input_dim = self.phobert_repr_dim + llm_proj_dim   # concat

        # ── Fusion MLP ────────────────────────────────────────────────────────
        self.fusion_head = FusionMLP(
            input_dim=fusion_input_dim,
            hidden_dim=512,
            num_labels=num_labels,
            dropout=fusion_dropout,
        )

        # ── SupCon Projection head ────────────────────────────────────────────
        if use_supcon:
            self.supcon_proj = SupConProjectionHead(input_dim=512, proj_dim=256)
        else:
            self.supcon_proj = None

        print(
            f"[LLMFusedPhoBERT v2] {model_name} ({self.phobert_hidden_size}H)\n"
            f"  Pooling: {'Hybrid(CLS+mean+max)' if use_hybrid_pooling else 'CLS-only'}\n"
            f"  Fusion : {'Gated interaction' if use_cross_attention else 'Concat'}\n"
            f"  SupCon : {use_supcon} (w={supcon_weight})\n"
            f"  Focal γ: {focal_gamma} | LS: {label_smoothing}\n"
            f"  LogitAdj τ: {logit_adjustment_tau}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) PhoBERT input token IDs
            attention_mask: (B, T) attention mask
            llm_embeddings: (B, llm_hidden_size) pre-extracted LLM embeddings
            labels: (B,) integer class labels (optional)

        Returns:
            Dict với keys: logits, loss (if labels), phobert_repr, llm_proj_output
        """
        # ── PhoBERT branch ────────────────────────────────────────────────────
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # (B, T, 1024)

        if self.use_hybrid_pooling:
            phobert_repr = self.phobert_pooling(last_hidden, attention_mask)  # (B, 1024)
        else:
            phobert_repr = self.phobert_dropout(last_hidden[:, 0, :])         # (B, 1024)

        # ── LLM branch ────────────────────────────────────────────────────────
        llm_proj = self.llm_proj(llm_embeddings)  # (B, 512)

        # ── Fusion ────────────────────────────────────────────────────────────
        if self.use_cross_attention and self.cross_attn_fusion is not None:
            fused = self.cross_attn_fusion(phobert_repr, llm_proj)  # (B, 1024)
        else:
            fused = torch.cat([phobert_repr, llm_proj], dim=-1)     # (B, 1536)

        logits, hidden = self.fusion_head(fused)   # (B, 3), (B, 512)

        # ── Logit Adjustment (at inference only — training uses raw logits) ──
        # Điều chỉnh logit dựa trên class prior để improve minority class recall
        adjusted_logits = logits
        if self.log_prior is not None and not self.training:
            adjusted_logits = apply_logit_adjustment(
                logits, self.log_prior, self.logit_adjustment_tau
            )

        result = {
            "logits": adjusted_logits,
            "phobert_repr": phobert_repr,
            "llm_proj_output": llm_proj,
            "hidden": hidden,
        }

        if labels is not None:
            # Compute SupCon features
            supcon_feats = None
            if self.use_supcon and self.supcon_proj is not None:
                supcon_feats = self.supcon_proj(hidden)  # (B, 256)

            result["loss"] = self._compute_loss(logits, labels, supcon_feats)

        return result

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        supcon_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Focal Loss + SupCon Loss (correctly computed)."""
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(logits.device)

        if self.use_focal_loss:
            task_loss = focal_loss(
                logits, labels,
                weight=weight,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
                num_classes=self.num_labels,
            )
        else:
            task_loss = F.cross_entropy(
                logits, labels,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )

        if self.use_supcon and supcon_features is not None:
            sc_loss = supcon_loss(
                supcon_features, labels,
                temperature=self.supcon_temperature,
            )
            return task_loss + self.supcon_weight * sc_loss

        return task_loss

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        phobert = sum(p.numel() for p in self.backbone.parameters())
        fusion = (
            sum(p.numel() for p in self.llm_proj.parameters())
            + sum(p.numel() for p in self.fusion_head.parameters())
            + (sum(p.numel() for p in self.supcon_proj.parameters()) if self.supcon_proj else 0)
            + (sum(p.numel() for p in self.cross_attn_fusion.parameters()) if self.cross_attn_fusion else 0)
            + (sum(p.numel() for p in self.phobert_pooling.parameters()) if self.use_hybrid_pooling else 0)
        )
        return {
            "total": total,
            "trainable": trainable,
            "phobert_backbone": phobert,
            "fusion_heads": fusion,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class LLMEmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset tích hợp text tokens (cho PhoBERT) và LLM embeddings đã trích xuất.
    LLM embeddings được load từ .npy file và cache trong RAM.
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        llm_embedding_path: str,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load LLM embeddings vào RAM
        self.llm_embeddings = np.load(llm_embedding_path).astype(np.float32)
        assert len(self.llm_embeddings) == len(texts), (
            f"LLM embedding count ({len(self.llm_embeddings)}) "
            f"!= text count ({len(texts)}). "
            f"Kiểm tra lại {llm_embedding_path}."
        )
        print(
            f"  LLM embeddings loaded: {self.llm_embeddings.shape} "
            f"from {llm_embedding_path}"
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "llm_embedding": torch.tensor(self.llm_embeddings[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_fusion_model_from_config(
    config: dict,
    class_prior: Optional[List[float]] = None,
) -> LLMFusedPhoBERT:
    """Build LLMFusedPhoBERT từ config dict."""
    model_cfg = config.get("model", {})
    fusion_cfg = config.get("fusion", {})
    training_cfg = config.get("training", {})

    return LLMFusedPhoBERT(
        model_name=model_cfg.get("name", "vinai/phobert-large"),
        num_labels=model_cfg.get("num_labels", 3),
        llm_hidden_size=fusion_cfg.get("llm_hidden_size", 896),
        llm_proj_dim=fusion_cfg.get("llm_proj_dim", 512),
        phobert_dropout=model_cfg.get("dropout", 0.1),
        fusion_dropout=fusion_cfg.get("fusion_dropout", 0.2),
        use_focal_loss=training_cfg.get("use_focal_loss", True),
        focal_gamma=training_cfg.get("focal_gamma", 3.0),
        label_smoothing=training_cfg.get("label_smoothing", 0.05),
        use_supcon=training_cfg.get("use_supcon", True),
        supcon_weight=training_cfg.get("supcon_weight", 0.1),
        supcon_temperature=training_cfg.get("supcon_temperature", 0.07),
        use_cross_attention=fusion_cfg.get("use_cross_attention", True),
        use_hybrid_pooling=fusion_cfg.get("use_hybrid_pooling", True),
        logit_adjustment_tau=training_cfg.get("logit_adjustment_tau", 0.3),
        class_prior=class_prior,
    )
