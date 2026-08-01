"""Tests for sparse concept inversion (Adam + FISTA) and the Mahalanobis prior."""
import numpy as np
import torch

from steerable_retrieval.steer.inversion import (
    MahalanobisPrior,
    fit_mahalanobis_prior,
    invert_concept,
    load_default_prior,
    mahalanobis_distance,
)


def test_prior_shapes_and_psd(audio, dims):
    prior = fit_mahalanobis_prior(audio)
    assert prior.mean.shape == (1, dims["D"])
    assert prior.precision.shape == (dims["D"], dims["D"])
    # precision is symmetric positive-definite
    P = prior.precision
    assert torch.allclose(P, P.T, atol=1e-5)
    assert torch.linalg.eigvalsh(P).min() > 0


def test_mahalanobis_distance_nonnegative(audio, prior):
    x = torch.from_numpy(audio[:16])
    d = mahalanobis_distance(x, prior)
    assert d.shape == (16,)
    assert (d >= 0).all()  # squared distance under a PSD precision is non-negative
    # a deliberately off-manifold direction is farther than an in-distribution sample
    far = torch.randn(1, prior.mean.shape[-1]) * 10.0
    assert float(mahalanobis_distance(far, prior).item()) > float(d.min().item())


def test_fista_reconstructs_direction(model, prior):
    """FISTA should recover a sparse code whose decode aligns with the concept text."""
    text = model.text_encoder(["loud distorted guitar"])
    res = invert_concept(model, text, prior, method="fista", num_steps=150)
    assert res.method == "fista"
    assert res.sparse_values.shape[0] == model.sae_encoder.dict_size
    assert res.support_mask.sum() > 0
    assert (res.sparse_values >= 0).all()  # non-negative code
    assert res.final_text_cosine > 0.5


def test_adam_runs_and_returns_support(model, prior, audio):
    text = model.text_encoder(["mellow piano"])
    res = invert_concept(model, text, prior, method="adam", audio_embeddings=audio, num_steps=40)
    assert res.method == "adam"
    assert res.sparse_values.shape[0] == model.sae_encoder.dict_size
    assert torch.isfinite(res.sparse_values).all()
    assert res.support_mask.sum() > 0


def test_unknown_method_raises(model, prior):
    text = model.text_encoder(["anything"])
    try:
        invert_concept(model, text, prior, method="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_prior_to_device_roundtrip(prior):
    moved = prior.to(device="cpu", dtype=torch.float64)
    assert moved.mean.dtype == torch.float64
    assert isinstance(moved, MahalanobisPrior)


def test_prior_save_load_roundtrip(tmp_path, prior):
    p = tmp_path / "prior.npz"
    prior.save(p, meta={"model": "test", "n": 3})
    loaded = MahalanobisPrior.load(p)
    assert torch.allclose(loaded.mean, prior.mean, atol=1e-5)
    assert torch.allclose(loaded.precision, prior.precision, atol=1e-5)


def test_default_prior_is_packaged():
    """The library ships a fixed manifold prior (MuQ / Music4All, 512-d)."""
    p = load_default_prior()
    assert p.mean.shape == (1, 512)
    assert p.precision.shape == (512, 512)
    assert torch.allclose(p.precision, p.precision.T, atol=1e-4)
