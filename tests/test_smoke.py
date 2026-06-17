"""
tests/test_smoke.py

Smoke tests for TinyPhoBERT core components.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def student_cfg():
    with open("configs/student_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def distill_cfg():
    with open("configs/distillation_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def student_model(student_cfg):
    from models.student import build_student_from_config
    model = build_student_from_config(student_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def dummy_batch():
    bs, seq_len = 4, 32
    return {
        "input_ids": torch.randint(1, 1000, (bs, seq_len)),
        "attention_mask": torch.ones(bs, seq_len, dtype=torch.long),
        "labels": torch.tensor([0, 1, 2, 0]),
    }


# ────────────────────────────────────────────────────────────────────────────
# Student Model Tests
# ────────────────────────────────────────────────────────────────────────────

class TestTinyPhoBERT:

    def test_instantiation(self, student_model):
        """Student model can be instantiated."""
        assert student_model is not None

    def test_parameter_count(self, student_model):
        """Student has fewer params than teacher (~135M)."""
        n_params = student_model.count_parameters()
        # Should be between 30M and 60M
        assert 20_000_000 < n_params < 70_000_000, (
            f"Expected 20M-70M params, got {n_params:,}"
        )

    def test_forward_pass(self, student_model, dummy_batch):
        """Forward pass returns correct logits shape."""
        with torch.no_grad():
            out = student_model(
                dummy_batch["input_ids"],
                dummy_batch["attention_mask"],
            )
        assert "logits" in out
        assert out["logits"].shape == (4, 3), f"Expected (4, 3), got {out['logits'].shape}"

    def test_forward_with_labels(self, student_model, dummy_batch):
        """Forward pass computes CE loss when labels provided."""
        with torch.no_grad():
            out = student_model(
                dummy_batch["input_ids"],
                dummy_batch["attention_mask"],
                dummy_batch["labels"],
            )
        assert "loss" in out
        assert out["loss"].item() > 0

    def test_distill_outputs(self, student_model, dummy_batch):
        """Distillation outputs have correct structure."""
        with torch.no_grad():
            out = student_model(
                dummy_batch["input_ids"],
                dummy_batch["attention_mask"],
                return_distill_outputs=True,
            )
        assert "hidden_states" in out
        assert "attentions" in out
        assert "projected_hidden" in out

        # Check number of layers
        assert len(out["hidden_states"]) == 7  # 6 layers + embedding
        assert len(out["attentions"]) == 6
        assert len(out["projected_hidden"]) == 7

        # Check dimensions
        bs, seq_len = dummy_batch["input_ids"].shape
        assert out["projected_hidden"][0].shape == (bs, seq_len, 768), (
            f"Expected projection to 768, got {out['projected_hidden'][0].shape}"
        )

    def test_model_size_mb(self, student_model):
        """Model size is reasonable."""
        size = student_model.model_size_mb()
        assert 50 < size < 300, f"Expected 50-300 MB, got {size:.1f} MB"


# ────────────────────────────────────────────────────────────────────────────
# Distillation Loss Tests
# ────────────────────────────────────────────────────────────────────────────

class TestDistillationLoss:

    def test_logit_kd_loss(self):
        """KL divergence loss is non-negative."""
        from models.distillation import MultiLevelDistillationLoss
        loss_fn = MultiLevelDistillationLoss(alpha=1.0, beta=0.0, gamma=0.0)
        s_logits = torch.randn(4, 3)
        t_logits = torch.randn(4, 3)
        loss = loss_fn.logit_kd_loss(s_logits, t_logits)
        assert loss.item() >= 0

    def test_full_distillation_loss(self, student_model, dummy_batch):
        """Full distillation loss runs without errors."""
        from models.distillation import MultiLevelDistillationLoss

        bs, seq_len = dummy_batch["input_ids"].shape
        loss_fn = MultiLevelDistillationLoss(
            alpha=0.5, beta=0.1, gamma=0.1,
            use_logit_kd=True, use_hidden_kd=True, use_attention_kd=True,
        )

        with torch.no_grad():
            out = student_model(
                dummy_batch["input_ids"],
                dummy_batch["attention_mask"],
                return_distill_outputs=True,
            )

        teacher_logits = torch.randn(bs, 3)
        teacher_hidden = tuple(torch.randn(bs, seq_len, 768) for _ in range(13))
        teacher_att = tuple(torch.rand(bs, 12, seq_len, seq_len) for _ in range(12))

        losses = loss_fn(
            student_logits=out["logits"],
            teacher_logits=teacher_logits,
            labels=dummy_batch["labels"],
            student_hidden=out["hidden_states"],
            teacher_hidden=teacher_hidden,
            student_projected=out["projected_hidden"],
            student_attentions=out["attentions"],
            teacher_attentions=teacher_att,
            attention_mask=dummy_batch["attention_mask"],
        )

        assert "loss" in losses
        assert "loss_ce" in losses
        assert "loss_kd" in losses
        assert "loss_hidden" in losses
        assert "loss_att" in losses
        assert losses["loss"].item() > 0

    def test_no_distill_config(self, dummy_batch):
        """A1 config (no distillation) only uses CE loss."""
        from models.distillation import MultiLevelDistillationLoss
        loss_fn = MultiLevelDistillationLoss(
            alpha=0.0, beta=0.0, gamma=0.0,
            use_logit_kd=False, use_hidden_kd=False, use_attention_kd=False,
        )
        s_logits = torch.randn(4, 3)
        t_logits = torch.randn(4, 3)
        losses = loss_fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=dummy_batch["labels"],
        )
        # KD, hidden, att losses should be 0
        assert losses["loss_kd"].item() == 0.0
        assert losses["loss_hidden"].item() == 0.0
        assert losses["loss_att"].item() == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Metrics Tests
# ────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_compute_metrics(self):
        from utils.metrics import compute_metrics
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "macro_precision" in metrics
        assert "macro_recall" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1

    def test_perfect_predictions(self):
        from utils.metrics import compute_metrics
        y = [0, 1, 2, 0, 1, 2]
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)

    def test_class_weights(self):
        from utils.data_utils import get_class_weights
        labels = [0, 0, 0, 1, 1, 2]  # Imbalanced: 3, 2, 1
        weights = get_class_weights(labels, num_classes=3)
        assert weights.shape == (3,)
        # Weight for rare class (2) should be highest
        assert weights[2] > weights[0]


# ────────────────────────────────────────────────────────────────────────────
# Data Utils Tests
# ────────────────────────────────────────────────────────────────────────────

class TestDataUtils:

    def test_dataset_length(self):
        """HateSpeechDataset returns correct length."""
        from utils.data_utils import HateSpeechDataset

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                import torch
                return {
                    "input_ids": torch.randint(1, 100, (1, kwargs.get("max_length", 32))),
                    "attention_mask": torch.ones(1, kwargs.get("max_length", 32), dtype=torch.long),
                }

        texts = ["hello world", "xin chào", "hate speech"]
        labels = [0, 1, 2]
        ds = HateSpeechDataset(texts, labels, FakeTokenizer(), max_length=32)
        assert len(ds) == 3

    def test_label_mapping(self):
        from utils.data_utils import LABEL2ID, ID2LABEL
        assert LABEL2ID["CLEAN"] == 0
        assert LABEL2ID["OFFENSIVE"] == 1
        assert LABEL2ID["HATE"] == 2
        assert ID2LABEL[0] == "CLEAN"
