"""Tests for edit-mask construction and sparse steering ops."""
import torch

from steerable_retrieval.steer.steering import (
    build_edit_mask,
    idf_weights,
    normalize_bundle,
    steer_sparse,
)


def test_build_edit_mask_topk_and_norm(dims):
    M, K = dims["M"], dims["K"]
    vals = torch.zeros(M)
    vals[[3, 10, 50, 100, 200]] = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.7])
    mask = build_edit_mask(vals, M, K=3, norm_bundle="norm")
    assert mask.shape == (M,)
    nz = torch.nonzero(mask).flatten().tolist()
    assert set(nz) == {3, 200, 50}  # top-3 by |value|
    assert abs(float(mask.norm()) - 1.0) < 1e-5  # unit-norm edit direction


def test_build_edit_mask_empty_support(dims):
    mask = build_edit_mask(torch.zeros(dims["M"]), dims["M"], K=5)
    assert float(mask.abs().sum()) == 0.0


def test_steer_sign_semantics():
    sparse = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([0.0, 0.0, 1.0, 0.0])
    amp = steer_sparse(sparse, mask, alpha=2.0)
    assert float(amp[0, 2]) == 2.0  # amplify adds
    # suppression clamps at zero (non-negative codes)
    sup = steer_sparse(sparse, mask, alpha=-5.0)
    supp_on_active = steer_sparse(torch.tensor([[0.0, 0.0, 1.0, 0.0]]), mask, alpha=-5.0)
    assert float(supp_on_active[0, 2]) == 0.0
    assert (sup >= 0).all()


def test_normalize_bundle_modes():
    w = torch.tensor([3.0, 4.0])
    assert abs(float(normalize_bundle(w, "norm").norm()) - 1.0) < 1e-6
    assert abs(float(normalize_bundle(w, "prob").abs().sum()) - 1.0) < 1e-6


def test_idf_downweights_frequent_features():
    codes = torch.zeros(100, 5)
    codes[:, 0] = 1.0       # feature 0 active everywhere -> low IDF
    codes[:3, 1] = 1.0      # feature 1 rare -> high IDF
    w = idf_weights(codes)
    assert w[1] > w[0]
