import math
import torch
from src.model.value_dist import bucket_centers, hl_gauss_target, expected_value


def test_bucket_centers():
    c = bucket_centers(4)
    assert torch.allclose(c, torch.tensor([0.125, 0.375, 0.625, 0.875]))


def test_hl_gauss_target_sums_to_one_and_peaks_at_v():
    v = torch.tensor([0.2, 0.8])
    t = hl_gauss_target(v, buckets=64, sigma_frac=0.75)
    assert t.shape == (2, 64)
    assert torch.allclose(t.sum(dim=1), torch.ones(2), atol=1e-5)
    assert t[0].argmax().item() in (12, 13)
    assert t[1].argmax().item() in (50, 51)


def test_expected_value_recovers_v():
    v = torch.tensor([0.3, 0.6, 0.9])
    t = hl_gauss_target(v, buckets=128, sigma_frac=0.75)
    vhat = expected_value(torch.log(t + 1e-12))
    assert torch.allclose(vhat, v, atol=1.0 / 128 * 2)
