"""
models/fusion_model.py

LLM-Fused PhoBERT for Vietnamese Hate Speech Detection.

Kiến trúc:
    Input Vietnamese comment
            │
            ├──────────────────────────────────┐
            ▼                                  ▼
    PhoBERT-large (trainable)         Qwen2.5-0.5B (frozen, offline)
            │                                  │
      CLS pooling (1024-dim)          pre-extracted .npy
            │                         LLM projection: 896→512
            │                                  │
            └──────────── concat ──────────────┘
                                │
                        (1024 + 512 = 1536)
                                │
                          Fusion MLP:
                          Linear(1536→512) → LayerNorm → GELU → Dropout
                          Linear(512→3)
                                │
                      CLEAN / OFFENSIVE / HATE

Novelty so với baseline PhoBERT-only:
    PhoBERT nắm rõ đặc điểm ngôn ngữ tiếng Việt (morphology, word order,
    teencode sau preprocessing), nhưng bị giới hạn bởi pretraining corpus
    (20GB ViNews + Wikipedia — formal text). Qwen2.5-0.5B được train trên
    diverse multilingual corpus bao gồm cả informal social media text, nên
    mang thêm semantic context về hate speech patterns mà PhoBERT chưa thấy.
    Fusion của 2 complementary representations → cải thiện phân biệt
    OFFENSIVE/HATE, đặc biệt với các câu ngắn/implicit hate.

Reference:
    - SHIELD: "Towards Interpretable Hate Speech Detection using LLM-extracted
      Rationales" (WOAH/ACL 2024)
    - LLMEmbed: "Rethinking Lightweight LLM's Genuine Function in Text
      Classification" (ACL 2024)
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class LLMProjectionHead(nn.Module):
    """
    Project LLM embedding (896-dim) sang fusion space (512-dim).
    Thêm LayerNorm sau projection để scale embedding từ LLM về
    cùng magnitude với PhoBERT CLS output — quan trọng khi concat.
    """

    def __init__(self, llm_hidden_size: int = 896, proj_dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Linear(llm_hidden_size, proj_dim, bias=True)
        self.norm = nn.LayerNorm(proj_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.proj(x)))


class FusionMLP(nn.Module):
    """
    MLP fusion head: concat(PhoBERT_CLS, LLM_proj) → logits.
    Dùng 2 layer để học cách kết hợp 2 không gian biểu diễn khác nhau.
    """

    def __init__(
        self,
        input_dim: int,       # 1024 + 512 = 1536
        hidden_dim: int = 512,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LLMFusedPhoBERT(nn.Module):
    """
    PhoBERT-large + Frozen LLM embedding fusion model.

    Args:
        model_name: PhoBERT model name (default: vinai/phobert-large)
        num_labels: Number of classification labels (default: 3)
        llm_hidden_size: Hidden size của LLM extractor (Qwen2.5-0.5B: 896)
        llm_proj_dim: Projection dimension cho LLM embedding (default: 512)
        phobert_dropout: Dropout trước CLS output của PhoBERT
        fusion_dropout: Dropout trong FusionMLP
        class_weights: Optional class weights cho Focal/Weighted CE loss
        use_focal_loss: Dùng Focal Loss thay vì CE
        focal_gamma: γ cho Focal Loss
        label_smoothing: Label smoothing
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
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # ── PhoBERT backbone ──────────────────────────────────────────────────
        config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.phobert_hidden_size = config.hidden_size  # 1024 for phobert-large
        self.num_layers = config.num_hidden_layers
        self.phobert_dropout = nn.Dropout(phobert_dropout)

        # ── LLM Projection head ───────────────────────────────────────────────
        self.llm_proj = LLMProjectionHead(llm_hidden_size, llm_proj_dim)

        # ── Fusion MLP ────────────────────────────────────────────────────────
        fusion_input_dim = self.phobert_hidden_size + llm_proj_dim
        self.fusion_head = FusionMLP(
            input_dim=fusion_input_dim,
            hidden_dim=512,
            num_labels=num_labels,
            dropout=fusion_dropout,
        )

        print(
            f"[LLMFusedPhoBERT] {model_name} ({self.phobert_hidden_size}H) "
            f"+ LLM_proj ({llm_proj_dim}H) → fusion ({fusion_input_dim}H) → {num_labels}"
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
                            đã được load từ .npy file, dtype=float32
            labels: (B,) integer class labels (optional)

        Returns:
            Dict với keys: logits, loss (if labels), cls_output, llm_proj_output
        """
        # ── PhoBERT branch ────────────────────────────────────────────────────
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token = first token của last hidden state
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
        cls_output = self.phobert_dropout(cls_output)

        # ── LLM branch ────────────────────────────────────────────────────────
        llm_proj = self.llm_proj(llm_embeddings)  # (B, 512)

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = torch.cat([cls_output, llm_proj], dim=-1)  # (B, 1536)
        logits = self.fusion_head(fused)                    # (B, 3)

        result = {
            "logits": logits,
            "cls_output": cls_output,
            "llm_proj_output": llm_proj,
            "fused": fused,
        }

        if labels is not None:
            result["loss"] = self._compute_loss(logits, labels)

        return result

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(logits.device)

        if self.use_focal_loss:
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

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        phobert = sum(p.numel() for p in self.backbone.parameters())
        fusion = sum(p.numel() for p in self.llm_proj.parameters()) + \
                 sum(p.numel() for p in self.fusion_head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "phobert_backbone": phobert,
            "fusion_heads": fusion,
        }


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

        # Load LLM embeddings vào RAM (Qwen2.5-0.5B: ~2.8MB / 1000 samples)
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


def build_fusion_model_from_config(config: dict) -> LLMFusedPhoBERT:
    """Build LLMFusedPhoBERT từ config dict."""
    model_cfg = config.get("model", {})
    fusion_cfg = config.get("fusion", {})
    return LLMFusedPhoBERT(
        model_name=model_cfg.get("name", "vinai/phobert-large"),
        num_labels=model_cfg.get("num_labels", 3),
        llm_hidden_size=fusion_cfg.get("llm_hidden_size", 896),
        llm_proj_dim=fusion_cfg.get("llm_proj_dim", 512),
        phobert_dropout=model_cfg.get("dropout", 0.1),
        fusion_dropout=fusion_cfg.get("fusion_dropout", 0.2),
        use_focal_loss=config.get("training", {}).get("use_focal_loss", True),
        focal_gamma=config.get("training", {}).get("focal_gamma", 2.0),
        label_smoothing=config.get("training", {}).get("label_smoothing", 0.1),
    )