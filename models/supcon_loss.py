"""
models/supcon_loss.py

Supervised Contrastive Loss (Khosla et al., 2020) cho Vietnamese Hate
Speech Detection.

Ý tưởng: bên cạnh Focal/CE loss tối ưu decision boundary, SupCon loss
hoạt động trực tiếp trên không gian embedding (pooled_output của PhoBERT,
TRƯỚC classifier head) — kéo các sample cùng label lại gần nhau, đẩy các
sample khác label ra xa.

Vì sao phù hợp với severe class imbalance (CLEAN 83%, OFFENSIVE 7%, HATE 11%):
    - SupCon hoạt động theo CẶP (pairwise) trong mỗi batch, không phụ thuộc
      tổng số sample mỗi class trong toàn bộ dataset.
    - Dù OFFENSIVE/HATE chỉ chiếm vài sample trong 1 batch, chúng vẫn nhận
      đủ gradient signal để co cụm lại — Cross-Entropy thuần không làm
      được điều này vì CE chỉ quan tâm đúng/sai ở quyết định cuối, không
      quan tâm cấu trúc hình học của embedding space.

Tham khảo: Khosla, P. et al. "Supervised Contrastive Learning." NeurIPS 2020.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Args:
        temperature: Nhiệt độ τ cho similarity scaling. Giá trị nhỏ (0.05-0.1)
                     làm loss "sắc" hơn — phạt nặng các cặp gần nhưng khác
                     label. Mặc định 0.07 theo paper gốc.
        base_temperature: Temperature dùng để chuẩn hóa scale của loss,
                           thường = temperature.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) embedding vectors — pooled_output của PhoBERT,
                      TRƯỚC dropout/classifier. KHÔNG cần normalize trước,
                      hàm tự L2-normalize.
            labels: (B,) integer class labels.

        Returns:
            Scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            # Không đủ sample để tạo cặp positive/negative
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2-normalize embeddings — bắt buộc để dot product = cosine similarity
        features = F.normalize(features, p=2, dim=1)

        labels = labels.contiguous().view(-1, 1)
        # mask[i,j] = 1 nếu sample i và j cùng label (và i != j)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Similarity matrix, scaled bởi temperature
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature

        # Trừ max mỗi hàng để ổn định số học (giống softmax trick chuẩn)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Loại bỏ self-similarity (i == j) khỏi cả positive mask lẫn denominator
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Denominator: tổng exp similarity với MỌI sample khác (trừ chính nó)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Chỉ những anchor có ít nhất 1 positive trong batch mới đóng góp loss
        # (tránh chia cho 0 nếu 1 class chỉ xuất hiện đúng 1 lần trong batch)
        num_positives = mask.sum(dim=1)
        valid = num_positives > 0

        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / num_positives[valid]

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)


class SupConProjectionHead(nn.Module):
    """
    Projection head nhỏ (tùy chọn) để map pooled_output → không gian
    contrastive riêng, tách biệt với không gian dùng cho classification.

    Theo SimCLR/SupCon gốc, dùng projection head (thay vì dùng thẳng
    pooled_output) thường cho kết quả tốt hơn vì nó cho phép embedding
    "chính" (dùng để classify) không bị ép buộc quá cứng theo cấu trúc
    contrastive — projection head hấp thụ phần biến dạng đó.

    Có thể bỏ qua (dùng thẳng pooled_output) nếu muốn pipeline đơn giản hơn,
    đánh đổi lấy có thể kém hiệu quả hơn một chút.
    """

    def __init__(self, input_dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)