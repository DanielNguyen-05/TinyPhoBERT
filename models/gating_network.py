"""
models/gating_network.py

Learned Gating Network — thay thế trọng số cố định (grid search trên val)
bằng 1 mạng nhỏ HỌC ĐƯỢC cách kết hợp N expert theo TỪNG câu cụ thể.

Input:  concat xác suất của N expert (N × num_labels chiều)
        Ví dụ: 5 experts × 3 classes = 15-dim vector
Output: N trọng số gating (softmax, tổng = 1) — áp dụng cho từng sample
        riêng biệt, KHÔNG phải 1 hằng số toàn cục như weighted-average
        tuyến tính trước đây.

Công thức:
    gate_weights = softmax(MLP(concat(p_1, p_2, ..., p_N)))     # (B, N)
    final_probs  = Σ_i gate_weights[:, i] * p_i                  # (B, num_labels)

Vì sao đây là kiến trúc thật (không chỉ "ensemble khác cách tính"):
    - Trọng số PHỤ THUỘC INPUT — câu khác nhau có thể được kết hợp khác nhau
      (ví dụ: câu nhiều teencode → gate tăng trọng số ViSoBERT tự động)
    - Có tham số học được (MLP), train bằng gradient descent qua CE/Focal
      loss trên nhãn thật — không phải chỉ post-hoc weight search.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    MLP nhỏ nhận concat-probabilities của N expert, học cách sinh trọng số
    kết hợp riêng cho từng sample.

    Args:
        n_experts: Số lượng expert model (ví dụ 5).
        n_classes: Số lượng class đầu ra (3: CLEAN/OFFENSIVE/HATE).
        hidden_dim: Kích thước hidden layer của gating MLP.
        dropout: Dropout trong gating MLP — tránh overfit vì input chỉ
                 15-dim, rất dễ overfit nếu MLP quá mạnh so với dữ liệu.
    """

    def __init__(
        self,
        n_experts: int = 5,
        n_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_classes = n_classes

        input_dim = n_experts * n_classes
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts),
        )

        # Khởi tạo nhẹ nhàng: gần với "trọng số đều nhau" lúc bắt đầu train,
        # giúp training ổn định hơn thay vì khởi đầu quá lệch về 1 expert.
        nn.init.xavier_uniform_(self.gate_net[0].weight, gain=0.5)
        nn.init.zeros_(self.gate_net[0].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, expert_probs: torch.Tensor) -> dict:
        """
        Args:
            expert_probs: (B, n_experts, n_classes) — xác suất softmax của
                          từng expert cho từng sample trong batch.

        Returns:
            dict với:
                gate_weights: (B, n_experts) — trọng số kết hợp, tổng=1 mỗi sample
                final_probs:  (B, n_classes) — xác suất cuối cùng sau kết hợp
                final_logits: (B, n_classes) — log(final_probs), dùng cho CE loss
        """
        batch_size = expert_probs.shape[0]
        concat_probs = expert_probs.reshape(batch_size, -1)  # (B, n_experts*n_classes)

        gate_logits = self.gate_net(concat_probs)             # (B, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)         # (B, n_experts)

        # Weighted combination: (B, n_experts, 1) * (B, n_experts, n_classes) → sum over experts
        weighted = gate_weights.unsqueeze(-1) * expert_probs   # (B, n_experts, n_classes)
        final_probs = weighted.sum(dim=1)                      # (B, n_classes)

        # Log cho CE loss — clamp tránh log(0)
        final_logits = torch.log(final_probs.clamp(min=1e-8))

        return {
            "gate_weights": gate_weights,
            "final_probs": final_probs,
            "final_logits": final_logits,
        }

    def compute_loss(
        self,
        expert_probs: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> dict:
        """Focal Loss trên final_logits (log của xác suất kết hợp)."""
        outputs = self.forward(expert_probs)
        final_logits = outputs["final_logits"]

        log_probs = F.log_softmax(final_logits, dim=-1)  # re-normalize an toàn số học
        n_classes = log_probs.shape[-1]

        # F.nll_loss KHÔNG hỗ trợ label_smoothing (chỉ F.cross_entropy có) —
        # tự implement label smoothing thủ công theo công thức chuẩn:
        #   loss = (1-eps)*NLL(target) + eps*mean(-log_probs mọi class)
        nll_per_class = -log_probs  # (B, n_classes)
        nll_target = nll_per_class.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)
        smooth_term = nll_per_class.mean(dim=1)  # (B,)
        ce = (1.0 - label_smoothing) * nll_target + label_smoothing * smooth_term

        if class_weights is not None:
            sample_weights = class_weights[labels]
            ce = ce * sample_weights

        with torch.no_grad():
            pt = torch.exp(-nll_target)
        focal_w = (1.0 - pt) ** focal_gamma
        loss = (focal_w * ce).mean()

        outputs["loss"] = loss
        return outputs