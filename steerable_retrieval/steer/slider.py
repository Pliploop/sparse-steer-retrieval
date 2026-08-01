"""The :class:`Slider` -- the public entry point for sparse steerable retrieval.

A ``Slider`` binds a free-form text concept to a sparse edit direction in a trained
SAE, obtained by sparse inversion (:mod:`steerable_retrieval.steer.inversion`). Once
built it can steer any query embedding along the concept axis and retrieve over a
corpus::

    slider = Slider("distorted guitar", model=model)   # uses the packaged manifold prior
    z_edited = slider.steer(z, alpha=1.0)               # alpha < 0 suppresses
    idx, scores = slider.retrieve(z, corpus, alpha=1.0, k=10)

The Mahalanobis **manifold prior** is a fixed distributional constant shipped with the
library (fit on Music4All MuQ-MuLan embeddings) -- it is *independent of the retrieval
corpus*. To adapt it to your own data, call :meth:`Slider.fit_prior` explicitly.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from steerable_retrieval.steer.inversion import (
    InversionResult,
    MahalanobisPrior,
    fit_mahalanobis_prior,
    invert_concept,
    load_default_prior,
)
from steerable_retrieval.steer.retrieval import topk_cosine_neighbors
from steerable_retrieval.steer.steering import build_edit_mask, decode_normalized, steer_sparse

# Filled once the main MuQ/music4all SAE is trained and uploaded to HuggingFace.
DEFAULT_MODEL_CLASS: Optional[str] = None
DEFAULT_CHECKPOINT: Optional[str] = None
# Name of the packaged manifold prior loaded when no prior is supplied.
DEFAULT_PRIOR_NAME = "muq_mulan_music4all"

ArrayLike = Union[np.ndarray, torch.Tensor]


def _resolve_model(model, model_class, checkpoint):
    if model is not None:
        return model
    if model_class is None and checkpoint is None and DEFAULT_CHECKPOINT is None:
        raise ValueError(
            "No model provided and no default checkpoint is available yet. "
            "Pass an explicit `model=` (a trained SAE with .sae_encoder/.sae_decoder/"
            ".text_encoder), or wait for the released pretrained checkpoint."
        )
    from steerable_retrieval.steer.loading import load_steerable_sae

    return load_steerable_sae(
        checkpoint or DEFAULT_CHECKPOINT, model_class=model_class or DEFAULT_MODEL_CLASS
    )


class Slider:
    """A concept slider over a trained SAE's sparse feature space."""

    def __init__(
        self,
        concept: str,
        *,
        model=None,
        model_class: Optional[str] = DEFAULT_MODEL_CLASS,
        checkpoint: Optional[str] = DEFAULT_CHECKPOINT,
        prior: Optional[MahalanobisPrior] = None,
        audio_embeddings: Optional[ArrayLike] = None,
        method: str = "adam",
        K: Optional[int] = None,
        idf: Optional[torch.Tensor] = None,
        norm_bundle: str = "norm",
        device: Optional[Union[str, torch.device]] = None,
        activation_threshold: float = 1e-6,
        prior_name: str = DEFAULT_PRIOR_NAME,
        **inversion_kwargs,
    ):
        """Build a slider for ``concept``.

        Args:
            concept: free-form text concept, e.g. ``"distorted guitar"``.
            model: a trained SAE exposing ``sae_encoder``, ``sae_decoder``, ``text_encoder``.
            model_class, checkpoint: alternative to ``model`` -- resolve a pretrained SAE.
            prior: an explicit :class:`MahalanobisPrior`. If omitted, the packaged
                default prior (``prior_name``) is used -- a fixed manifold constant,
                *not* fit from any corpus. Use :meth:`fit_prior` to adapt it.
            audio_embeddings: optional ``[N, d]`` audio embeddings used *only* to
                initialize Adam inversion from the nearest audio neighbour. They do
                **not** set the prior.
            method: ``"adam"`` (default) or ``"fista"``.
            K: cap the concept support to the top-``K`` features (``None`` = all active).
            idf: optional IDF weights over dictionary features (down-weights hub neurons).
            norm_bundle: edit-direction normalization (``"norm"`` = unit L2, recommended).
            prior_name: which packaged prior to load when ``prior`` is omitted.
        """
        self.concept = concept
        self.method = method
        self.K = K
        self.model = _resolve_model(model, model_class, checkpoint)
        self.device = torch.device(device) if device is not None else self._infer_device()
        self._idf = idf
        self._norm_bundle = norm_bundle
        self._activation_threshold = activation_threshold
        self._inversion_kwargs = inversion_kwargs
        self._audio_np = _to_numpy(audio_embeddings) if audio_embeddings is not None else None

        # Prior precedence: explicit > packaged default constant. (audio_embeddings
        # is an init aid only and never becomes the prior -- the prior is decoupled
        # from any corpus. Call fit_prior() to adapt it on purpose.)
        if prior is not None:
            self.prior = prior.to(device=self.device)
        else:
            self.prior = load_default_prior(prior_name, device=self.device)
        self._check_prior_dim()

        self._text_embedding = self._embed_text(concept)
        self._recompute()

    # -- prior --------------------------------------------------------------- #
    def _check_prior_dim(self) -> None:
        d_model = int(getattr(self.model.sae_decoder, "act_size", self.model.sae_decoder.W_dec.shape[-1]))
        d_prior = int(self.prior.mean.shape[-1])
        if d_prior != d_model:
            raise ValueError(
                f"Manifold prior dim ({d_prior}) != model embedding dim ({d_model}). "
                "Pass a matching `prior=`, fit one via fit_prior(embeddings), or set the "
                "correct `prior_name`."
            )

    def fit_prior(self, embeddings: ArrayLike) -> "Slider":
        """Refit the manifold prior to ``embeddings`` and recompute the slider.

        Opt-in coupling: by default the slider uses the packaged prior, which is
        independent of any corpus. Pass your retrieval corpus (or any representative
        set) here to adapt the manifold prior to your own data. Returns ``self``.
        """
        self.prior = fit_mahalanobis_prior(_to_numpy(embeddings), device=self.device)
        self._check_prior_dim()
        self._recompute()
        return self

    def _recompute(self) -> None:
        """Re-run inversion + rebuild the edit mask (after a prior/param change)."""
        self.inversion: InversionResult = invert_concept(
            self.model,
            self._text_embedding,
            self.prior,
            method=self.method,
            audio_embeddings=self._audio_np,
            activation_threshold=self._activation_threshold,
            **self._inversion_kwargs,
        )
        self.mask = build_edit_mask(
            self.inversion.sparse_values,
            dict_size=self.model.sae_encoder.dict_size,
            K=self.K,
            idf=self._idf,
            norm_bundle=self._norm_bundle,
            activation_threshold=self._activation_threshold,
        )

    # -- introspection ------------------------------------------------------- #
    @property
    def support(self) -> torch.Tensor:
        """Indices of the active concept features (the slider's support)."""
        return torch.nonzero(self.mask.abs() > 0, as_tuple=False).flatten()

    def __len__(self) -> int:
        return int(self.support.numel())

    def __repr__(self) -> str:
        return (
            f"Slider(concept={self.concept!r}, method={self.method!r}, "
            f"|support|={len(self)}, text_cos={self.inversion.final_text_cosine:.3f})"
        )

    # -- steering ------------------------------------------------------------ #
    def steer(self, z: ArrayLike, alpha: float = 1.0) -> torch.Tensor:
        """Steer dense query embedding(s) ``z`` along the concept axis.

        ``alpha > 0`` amplifies the concept, ``alpha < 0`` suppresses it. Returns the
        L2-normalized edited dense embedding(s) with the same leading shape as ``z``.
        """
        sparse, squeeze = self._encode_sparse(z)
        mask = self.mask.to(device=sparse.device, dtype=sparse.dtype)
        edited = steer_sparse(sparse, mask, alpha)
        dense = decode_normalized(self.model, edited)
        return dense.squeeze(0) if squeeze else dense

    def amplify(self, z: ArrayLike, alpha: float = 1.0) -> torch.Tensor:
        return self.steer(z, abs(alpha))

    def suppress(self, z: ArrayLike, alpha: float = 1.0) -> torch.Tensor:
        return self.steer(z, -abs(alpha))

    # -- retrieval ----------------------------------------------------------- #
    def retrieve(
        self,
        z: ArrayLike,
        corpus: ArrayLike,
        *,
        alpha: float = 1.0,
        k: int = 10,
        exclude_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Steer ``z`` by ``alpha`` then return top-``k`` cosine neighbours in ``corpus``.

        ``corpus`` is an arbitrary set of dense embeddings to search over -- it is
        independent of the manifold prior. Returns ``(indices, scores)`` into ``corpus``.
        """
        edited = self.steer(z, alpha)
        gallery = _to_tensor(corpus, self.device)
        return topk_cosine_neighbors(edited.to(self.device), gallery, k, exclude_idx=exclude_idx)

    # -- internals ----------------------------------------------------------- #
    def _infer_device(self) -> torch.device:
        return self.model.sae_decoder.W_dec.device

    def _embed_text(self, concept: str) -> torch.Tensor:
        with torch.inference_mode():
            emb = self.model.text_encoder([concept])
        return emb.detach().clone().to(self.device)

    def _encode_sparse(self, z: ArrayLike) -> Tuple[torch.Tensor, bool]:
        t = _to_tensor(z, self.device)
        squeeze = t.dim() == 1
        if squeeze:
            t = t.unsqueeze(0)
        with torch.inference_mode():
            _, sparse, _, _ = self.model.inference(t)
        return sparse.clone(), squeeze


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _to_tensor(x: ArrayLike, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
