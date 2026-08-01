"""End-to-end tests for the Slider public API."""
import numpy as np
import pytest
import torch

from steerable_retrieval.steer import Slider


@pytest.mark.parametrize("method", ["adam", "fista"])
def test_slider_build_and_steer(model, prior, audio, dims, method):
    K = dims["K"]
    # prior is explicit (the synthetic model is 64-d; the packaged default is 512-d);
    # audio_embeddings only aids the Adam init.
    slider = Slider("distorted guitar", model=model, prior=prior, audio_embeddings=audio,
                    method=method, K=K, num_steps=40)
    assert 0 < len(slider) <= K
    assert set(slider.support.tolist()) == set(torch.nonzero(slider.mask).flatten().tolist())

    z = audio[0]
    edited = slider.steer(z, alpha=1.0)
    assert edited.shape == (dims["D"],)
    assert torch.isfinite(edited).all()
    assert abs(float(edited.norm()) - 1.0) < 1e-4  # retrieval embeddings are L2-normalized


def test_fit_prior_recomputes(model, prior, audio, dims):
    """fit_prior() adapts the manifold prior and recomputes the slider in place."""
    slider = Slider("cello", model=model, prior=prior, method="fista", K=dims["K"], num_steps=60)
    before = slider.mask.clone()
    ret = slider.fit_prior(audio)  # opt-in: adapt the prior to a corpus
    assert ret is slider
    assert slider.mask.shape == before.shape
    assert torch.isfinite(slider.mask).all()
    assert 0 < len(slider) <= dims["K"]


def test_alpha_monotonic_displacement(model, prior, audio, dims):
    slider = Slider("bright synths", model=model, prior=prior, method="fista",
                    K=dims["K"], num_steps=80)
    z = audio[1]
    z0 = slider.steer(z, alpha=0.0)
    near = slider.steer(z, alpha=0.5)
    far = slider.steer(z, alpha=3.0)
    # larger |alpha| moves further from the alpha=0 point
    assert float(torch.dot(z0, far)) < float(torch.dot(z0, near))


def test_amplify_suppress_are_signed_steer(model, prior, audio, dims):
    slider = Slider("piano", model=model, prior=prior, method="fista", K=dims["K"], num_steps=80)
    z = audio[2]
    assert torch.allclose(slider.amplify(z, 1.5), slider.steer(z, 1.5))
    assert torch.allclose(slider.suppress(z, 1.5), slider.steer(z, -1.5))


def test_retrieve_excludes_and_sorts(model, prior, audio, dims):
    slider = Slider("guitar", model=model, prior=prior, method="fista", K=dims["K"], num_steps=80)
    idx, scores = slider.retrieve(audio[0], audio, alpha=1.0, k=5, exclude_idx=0)
    assert idx.shape[0] == 5
    assert 0 not in idx.tolist()
    assert scores[0] >= scores[-1]


def test_batched_steer(model, prior, audio, dims):
    slider = Slider("drums", model=model, prior=prior, method="fista", K=dims["K"], num_steps=80)
    out = slider.steer(audio[:8], alpha=1.0)
    assert out.shape == (8, dims["D"])


def test_model_reuse_across_concepts(model, prior, audio, dims):
    """Building many sliders should reuse one loaded model + prior (efficiency path)."""
    s1 = Slider("violin", model=model, prior=prior, method="fista", K=dims["K"], num_steps=60)
    s2 = Slider("trumpet", model=model, prior=prior, method="fista", K=dims["K"], num_steps=60)
    assert s1.model is s2.model
    # prior tensors are shared (not recomputed/recopied) across sliders on the same device
    assert s1.prior.mean is s2.prior.mean
    assert s1.prior.precision is s2.prior.precision
    assert s1.concept != s2.concept


def test_defaults_and_errors(model):
    # No prior given -> packaged default (512-d) is loaded, which mismatches the
    # 64-d synthetic model, so the dim check raises a helpful error.
    with pytest.raises(ValueError, match="prior dim"):
        Slider("x", model=model)
    # No model and no released checkpoint yet.
    with pytest.raises(ValueError):
        Slider("x")
