"""Sparse inversion for open-vocabulary concept attribution.

This is the core method of *"Steering dense music retrieval with open-vocabulary
concept discovery"* (ISMIR 2026). Given a free-form concept's text embedding
``z_c`` in a joint music--text space, we recover a **sparse code** whose decoded
audio embedding reconstructs ``z_c`` while staying close to the empirical audio
manifold (a Mahalanobis prior). Two solvers are provided:

* ``method="adam"``  -- gradient descent on the latent pre-activations through the
  SAE's own sparsifying operator (the general, differentiable variant).
* ``method="fista"`` -- a fast linear-inverse solver exploiting the linear decoder
  (~20 ms/inversion on CPU; used for the interactive demo).

The recovered support is the concept "slider": the set of sparse features that,
when amplified or suppressed, steer retrieval along the concept axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Mahalanobis audio-manifold prior
# --------------------------------------------------------------------------- #
@dataclass
class MahalanobisPrior:
    """Gaussian prior over L2-normalized audio embeddings.

    ``mean`` is ``[1, d]`` and ``precision`` is ``[d, d]`` (inverse covariance).
    Used to keep inverted concepts near the region occupied by real audio.
    """

    mean: torch.Tensor
    precision: torch.Tensor

    def to(self, device=None, dtype=None) -> "MahalanobisPrior":
        return MahalanobisPrior(
            mean=self.mean.to(device=device, dtype=dtype),
            precision=self.precision.to(device=device, dtype=dtype),
        )

    def save(self, path, *, meta: Optional[dict] = None) -> None:
        """Save the prior as a compact ``.npz`` (mean, precision, optional JSON meta)."""
        arrs = {
            "mean": self.mean.detach().cpu().numpy().astype(np.float32),
            "precision": self.precision.detach().cpu().numpy().astype(np.float32),
        }
        if meta:
            import json

            arrs["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(path, **arrs)

    @classmethod
    def load(cls, path, *, device=None, dtype: torch.dtype = torch.float32) -> "MahalanobisPrior":
        """Load a prior saved by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            mean = torch.from_numpy(data["mean"]).to(device=device, dtype=dtype)
            precision = torch.from_numpy(data["precision"]).to(device=device, dtype=dtype)
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        return cls(mean=mean, precision=precision)


def load_default_prior(name: str = "muq_mulan_music4all", *, device=None) -> MahalanobisPrior:
    """Load a Mahalanobis prior shipped with the package (a library constant).

    The default (``muq_mulan_music4all``) is fit once on Music4All MuQ-MuLan audio
    embeddings and packaged under ``steerable_retrieval/assets/``. It is a fixed
    distributional constant of the audio manifold — independent of whatever corpus
    a caller later retrieves over.
    """
    from importlib import resources

    res = resources.files("steerable_retrieval.assets").joinpath(f"{name}_prior.npz")
    with resources.as_file(res) as p:
        return MahalanobisPrior.load(p, device=device)


def fit_mahalanobis_prior(
    embeddings,
    *,
    cov_eps: float = 1e-3,
    shrinkage: float = 0.05,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> MahalanobisPrior:
    """Fit a shrinkage Mahalanobis prior to a set of audio embeddings.

    Args:
        embeddings: ``[N, d]`` array/tensor of audio embeddings (need not be normalized).
        cov_eps: diagonal jitter added to the covariance for invertibility.
        shrinkage: convex shrinkage toward a scaled identity (Ledoit-Wolf style).
    """
    x = np.asarray(
        embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings,
        dtype=np.float64,
    )
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    denom = max(int(centered.shape[0] - 1), 1)
    cov = (centered.T @ centered) / float(denom)
    trace_scale = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] > 0 else 1.0
    cov = (1.0 - shrinkage) * cov + shrinkage * trace_scale * np.eye(cov.shape[0])
    cov = cov + cov_eps * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov)
    return MahalanobisPrior(
        mean=torch.from_numpy(mean).to(device=device, dtype=dtype),
        precision=torch.from_numpy(precision).to(device=device, dtype=dtype),
    )


def mahalanobis_distance(x: torch.Tensor, prior: MahalanobisPrior) -> torch.Tensor:
    """Squared Mahalanobis distance of (L2-normalized) ``x`` to the prior, per row."""
    x = F.normalize(x, dim=-1)
    delta = x - prior.mean
    return torch.einsum("bi,ij,bj->b", delta, prior.precision, delta)


# --------------------------------------------------------------------------- #
# Re-applying the SAE's sparsifying operator Pi to arbitrary latent logits
# --------------------------------------------------------------------------- #
def apply_encoder_sparse_transform(
    model, latent_logits: torch.Tensor, use_encoder_penalty: bool = True
) -> torch.Tensor:
    """Apply the SAE encoder's sparsifying nonlinearity ``Pi`` to ``latent_logits``.

    This lets the Adam inversion optimize *pre-activations* while keeping the exact
    training-time sparsity structure of the model (TopK / BatchTopK / JumpReLU / ReLU).
    """
    encoder = model.sae_encoder
    encoder_name = type(encoder).__name__
    if encoder_name == "VanillaSAEEncoder":
        return F.relu(latent_logits)
    if encoder_name == "TopKSAEEncoder":
        acts = F.relu(latent_logits)
        return encoder.penalty(acts) if use_encoder_penalty else acts
    if encoder_name in {"BatchTopKSAEEncoder", "MatryoshkaBatchTopKSAEEncoder"}:
        acts = F.relu(latent_logits)
        if not use_encoder_penalty:
            return acts
        threshold = getattr(encoder, "running_threshold", None)
        if threshold is None:
            threshold = encoder.penalty._get_threshold(acts)
        return encoder.penalty(acts, threshold)
    if encoder_name == "JumpReLUSAEEncoder":
        return encoder.jumprelu(latent_logits) if use_encoder_penalty else F.relu(latent_logits)
    raise NotImplementedError(
        f"Latent-logit inversion is not implemented for encoder type {encoder_name}."
    )


# --------------------------------------------------------------------------- #
# FISTA solver for the linear inverse problem with a Mahalanobis term
# --------------------------------------------------------------------------- #
def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def fista_mahalanobis(
    A: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    M: torch.Tensor,
    lam: float,
    gamma: float,
    n_iter: int = 200,
    L: Optional[float] = None,
    power_iters: int = 30,
    enforce_nonnegative: bool = True,
) -> torch.Tensor:
    """Solve ``min_x 0.5||Ax - y||^2 + 0.5*gamma*(Ax-mu)^T M (Ax-mu) + lam||x||_1``.

    ``A`` is ``W_dec^T`` (``[d, m]``); ``x`` is the sparse code (``[m]``). Non-negativity
    matches ReLU-style SAE codes. ``L`` (Lipschitz constant) is estimated by power
    iteration on the Hessian ``A^T A + gamma A^T M A`` when not provided.
    """
    n, m = A.shape
    del n
    x = torch.zeros(m, device=A.device, dtype=A.dtype)
    z = x.clone()
    t = 1.0
    if L is None:
        v = torch.randn(m, device=A.device, dtype=A.dtype)
        v = v / v.norm().clamp_min(1e-12)
        for _ in range(power_iters):
            Hv = A.T @ (A @ v) + gamma * (A.T @ (M @ (A @ v)))
            v = Hv / Hv.norm().clamp_min(1e-12)
        Hv = A.T @ (A @ v) + gamma * (A.T @ (M @ (A @ v)))
        L = torch.dot(v, Hv)
    eta = 1.0 / float(torch.as_tensor(L).item())
    for _ in range(n_iter):
        Az = A @ z
        grad = A.T @ (Az - y) + gamma * (A.T @ (M @ (Az - mu)))
        x_new = soft_threshold(z - eta * grad, lam * eta)
        if enforce_nonnegative:
            x_new = F.relu(x_new)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


# --------------------------------------------------------------------------- #
# Inversion result + the two solvers + a dispatcher
# --------------------------------------------------------------------------- #
@dataclass
class InversionResult:
    """Output of a concept inversion.

    ``sparse_values`` is the recovered sparse code over the SAE dictionary ``[m]``;
    ``support_mask`` is the boolean support (active features). ``final_text_cosine``
    and ``final_mahalanobis_distance`` are diagnostics of solution quality.
    """

    sparse_values: torch.Tensor
    support_mask: torch.Tensor
    final_text_cosine: float
    final_mahalanobis_distance: float
    method: str


def _decoder_matrices(model, device, dtype):
    """Return ``A = W_dec^T`` (``[d, m]``) and ``b_dec`` (``[d]``)."""
    A = model.sae_decoder.W_dec.detach().to(device=device, dtype=dtype).T
    b_dec = model.sae_decoder.b_dec.detach().to(device=device, dtype=dtype).view(-1)
    return A, b_dec


def invert_concept_adam(
    model,
    text_embedding: torch.Tensor,
    prior: MahalanobisPrior,
    *,
    init_sparse: Optional[torch.Tensor] = None,
    num_steps: int = 500,
    lr: float = 1e-3,
    reg_strength: float = 1e-4,
    use_encoder_penalty: bool = True,
    activation_threshold: float = 1e-6,
) -> InversionResult:
    """Adam inversion: optimize latent pre-activations through the SAE operator ``Pi``.

    ``init_sparse`` (``[1, m]`` or ``[m]``) initializes the latent logits; the paper
    initializes from the sparse code of the nearest audio neighbour to ``z_c`` for
    stable convergence (see :func:`nearest_audio_sparse_init`).
    """
    device = text_embedding.device
    dtype = text_embedding.dtype
    prior = prior.to(device=device, dtype=dtype)
    if init_sparse is None:
        init = torch.zeros((1, model.sae_encoder.dict_size), device=device, dtype=dtype)
    else:
        init = init_sparse.view(1, -1).to(device=device, dtype=dtype)

    latent_logits = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.AdamW([latent_logits], lr=lr)
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        z_sparse = apply_encoder_sparse_transform(model, latent_logits, use_encoder_penalty)
        z_hat = model.sae_decoder(z_sparse)
        text_cos = F.cosine_similarity(z_hat, text_embedding, dim=-1).mean()
        mahal = mahalanobis_distance(z_hat, prior).mean()
        loss = (1.0 - text_cos) + reg_strength * mahal
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        z_sparse = apply_encoder_sparse_transform(model, latent_logits, use_encoder_penalty)
        z_hat = model.sae_decoder(z_sparse)
        sparse_values = z_sparse.squeeze(0).detach().cpu()
        support_mask = (sparse_values.abs() > activation_threshold).bool()
        final_cos = float(F.cosine_similarity(z_hat, text_embedding, dim=-1).mean().item())
        final_mahal = float(mahalanobis_distance(z_hat, prior).mean().item())
    return InversionResult(sparse_values, support_mask, final_cos, final_mahal, "adam")


def invert_concept_fista(
    model,
    text_embedding: torch.Tensor,
    prior: MahalanobisPrior,
    *,
    num_steps: int = 250,
    l1_lambda: float = 0.01,
    gamma: float = 1e-4,
    lipschitz: Optional[float] = None,
    power_iters: int = 30,
    enforce_nonnegative: bool = True,
    use_encoder_penalty: bool = True,
    apply_encoder_penalty_after: bool = True,
    activation_threshold: float = 1e-6,
) -> InversionResult:
    """FISTA inversion exploiting the linear decoder (fast; used by the live demo)."""
    device = text_embedding.device
    dtype = text_embedding.dtype
    prior = prior.to(device=device, dtype=dtype)
    A, b_dec = _decoder_matrices(model, device, dtype)
    y = text_embedding.squeeze(0).detach().to(device=device, dtype=dtype) - b_dec
    mu = prior.mean.squeeze(0).detach().to(device=device, dtype=dtype) - b_dec
    M = prior.precision.detach().to(device=device, dtype=dtype)

    x = fista_mahalanobis(
        A, y, mu, M, l1_lambda, gamma,
        n_iter=num_steps, L=lipschitz,
        power_iters=power_iters, enforce_nonnegative=enforce_nonnegative,
    )
    z_sparse = x.unsqueeze(0)
    if apply_encoder_penalty_after:
        z_sparse = apply_encoder_sparse_transform(model, z_sparse, use_encoder_penalty)

    with torch.no_grad():
        z_hat = model.sae_decoder(z_sparse)
        sparse_values = z_sparse.squeeze(0).detach().cpu()
        support_mask = (sparse_values.abs() > activation_threshold).bool()
        final_cos = float(F.cosine_similarity(z_hat, text_embedding, dim=-1).mean().item())
        final_mahal = float(mahalanobis_distance(z_hat, prior).mean().item())
    return InversionResult(sparse_values, support_mask, final_cos, final_mahal, "fista")


def nearest_audio_sparse_init(
    model,
    text_embedding: torch.Tensor,
    audio_embeddings: np.ndarray,
    audio_norms: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Sparse code of the audio embedding nearest (cosine) to ``text_embedding``.

    Recommended initialization for Adam inversion (paper Section 3.3).
    """
    text_norm = F.normalize(text_embedding.detach(), dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    audio_embeddings = np.asarray(audio_embeddings, dtype=np.float32)
    if audio_norms is None:
        audio_norms = np.clip(np.linalg.norm(audio_embeddings, axis=1), 1e-12, None)
    sims = (audio_embeddings @ text_norm) / audio_norms
    best = int(np.argmax(sims))
    device = model.sae_decoder.W_dec.device
    batch = torch.from_numpy(audio_embeddings[best][None]).to(device)
    with torch.inference_mode():
        _, z, _, _ = model.inference(batch)
    return z.detach().squeeze(0)


def prior_mean_sparse_init(model, prior: MahalanobisPrior) -> torch.Tensor:
    """Sparse code of the prior's mean embedding — a manifold-centered Adam init.

    Used when no ``audio_embeddings`` are available for a nearest-neighbour init, so
    Adam inversion still starts from a plausible audio point rather than zeros.
    """
    device = model.sae_decoder.W_dec.device
    mean = prior.mean.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        _, z, _, _ = model.inference(mean)
    return z.detach().squeeze(0)


def invert_concept(
    model,
    text_embedding: torch.Tensor,
    prior: MahalanobisPrior,
    *,
    method: str = "adam",
    audio_embeddings: Optional[np.ndarray] = None,
    **kwargs,
) -> InversionResult:
    """Dispatch to the requested inversion solver.

    For ``method="adam"`` the optimization is initialized from the nearest audio
    neighbour when ``audio_embeddings`` is provided (recommended by the paper), and
    otherwise from the prior mean — so inversion needs only the prior, not a corpus.
    """
    method = method.lower()
    if method == "adam":
        if kwargs.get("init_sparse") is None:
            kwargs["init_sparse"] = (
                nearest_audio_sparse_init(model, text_embedding, audio_embeddings)
                if audio_embeddings is not None
                else prior_mean_sparse_init(model, prior)
            )
        return invert_concept_adam(model, text_embedding, prior, **kwargs)
    if method == "fista":
        return invert_concept_fista(model, text_embedding, prior, **kwargs)
    raise ValueError(f"Unknown inversion method {method!r}; use 'adam' or 'fista'.")
