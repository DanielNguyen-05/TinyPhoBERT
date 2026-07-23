import torch
import numpy as np

from models.fusion_model import CrossAttentionFusion, apply_logit_adjustment
from class_aware_ensemble import (
    ClassAwareEnsemble,
    fit_class_aware_ensemble,
    temperature_scale_probs,
)


def test_fusion_is_conditioned_on_phobert_query():
    """Changing PhoBERT must change how the same LLM vector is fused."""
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(phobert_dim=8, llm_dim=4, num_heads=2, dropout=0.0)
    fusion.eval()

    llm = torch.randn(1, 4).repeat(2, 1)
    phobert = torch.stack([torch.ones(8), -torch.ones(8)])
    output = fusion(phobert, llm)

    residual_only_difference = fusion.norm(phobert)[0] - fusion.norm(phobert)[1]
    actual_difference = output[0] - output[1]
    assert not torch.allclose(actual_difference, residual_only_difference)


def test_log_prior_correction_favors_rare_class():
    prior = torch.tensor([0.8, 0.05, 0.15])
    corrected = apply_logit_adjustment(torch.zeros(3), torch.log(prior), tau=0.3)
    assert corrected[1] > corrected[2] > corrected[0]


def test_temperature_scaling_preserves_distribution_and_argmax():
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    calibrated = temperature_scale_probs(probs, 2.0)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    np.testing.assert_array_equal(calibrated.argmax(1), probs.argmax(1))
    assert calibrated[0, 0] < probs[0, 0]


def test_class_aware_ensemble_learns_class_specific_experts():
    rng = np.random.RandomState(7)
    labels = np.tile(np.arange(3), 40)
    probs = np.full((len(labels), 3, 3), 0.05)
    for row, label in enumerate(labels):
        # Expert m is reliable specifically for class m.
        probs[row, :, :] = rng.uniform(0.05, 0.25, size=(3, 3))
        probs[row, label, label] = 0.95
    probs /= probs.sum(axis=-1, keepdims=True)

    model = fit_class_aware_ensemble(
        probs, labels, temperatures=np.ones(3), regularization=0.01
    )
    assert isinstance(model, ClassAwareEnsemble)
    assert (model.class_weights.argmax(axis=0) == np.arange(3)).all()
    assert (model.predict_proba(probs).argmax(axis=1) == labels).mean() > 0.95
