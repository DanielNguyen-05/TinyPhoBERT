import torch

from models.fusion_model import CrossAttentionFusion, apply_logit_adjustment


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
