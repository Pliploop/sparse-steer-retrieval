"""Steering operations: turn an inverted concept support into an edit direction.

The concept support (from :mod:`steerable_retrieval.steer.inversion`) is converted to a masked,
optionally IDF-reweighted, L2-normalized direction over the SAE dictionary. Steering
adds (amplify) or subtracts (suppress) a scaled copy of that direction from a query's
sparse code, then decodes back to the dense space for retrieval.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def idf_weights(sparse_codes: torch.Tensor, *, eps: float = 1e-6, thresh: float = 1e-6) -> torch.Tensor:
    """IDF weights over dictionary features from a corpus of sparse codes.

    ``w_k = log(N / (df_k + eps) + 1)`` where ``df_k`` is the number of documents
    (audio samples) in which feature ``k`` is active. Down-weights generic hub neurons.
    """
    num_docs = sparse_codes.shape[0]
    df = (sparse_codes.abs() > thresh).float().sum(dim=0)
    return torch.log(torch.tensor(float(num_docs)) / (df + eps) + 1.0)


def apply_idf(values: torch.Tensor, idf: Optional[torch.Tensor]) -> torch.Tensor:
    values = values.detach().cpu().float().view(-1)
    if idf is not None and idf.shape[0] == values.shape[0]:
        values = values * idf.detach().cpu().float().view(-1)
    return values


def normalize_bundle(weights: torch.Tensor, mode: str = "norm", eps: float = 1e-6) -> torch.Tensor:
    """Normalize support weights so ``alpha`` is the sole edit-strength knob.

    ``mode="norm"`` (default) makes the edit a unit-L2 direction, so support size
    controls *which* coordinates move and ``alpha`` controls *how far*.
    """
    weights = weights.detach().cpu().float()
    if not mode or weights.numel() == 0:
        return weights
    if mode == "prob":
        total = float(weights.abs().sum().item())
        return weights if total <= eps else weights / total
    if mode == "normsum":
        total = float(weights.abs().sum().item())
        return weights / (total + eps) * float(weights.numel())
    weight_norm = float(weights.norm().item())
    return weights if weight_norm <= eps else weights / weight_norm


def build_edit_mask(
    sparse_values: torch.Tensor,
    dict_size: int,
    *,
    K: Optional[int] = None,
    idf: Optional[torch.Tensor] = None,
    norm_bundle: str = "norm",
    activation_threshold: float = 1e-6,
) -> torch.Tensor:
    """Build a dense ``[dict_size]`` edit direction from recovered sparse values.

    Selects the top-``K`` active features (by |value|, after optional IDF reweighting),
    normalizes them, and scatters them into a full-width mask.
    """
    values = apply_idf(sparse_values, idf)
    active = torch.where(values.abs() > activation_threshold)[0]
    if active.numel() == 0:
        return torch.zeros(dict_size, dtype=torch.float32)
    active_vals = values[active]
    k = active.numel() if K is None else min(int(K), int(active.numel()))
    order = torch.topk(active_vals.abs(), k=k).indices
    idx = active[order]
    vals = normalize_bundle(active_vals[order], mode=norm_bundle)
    mask = torch.zeros(dict_size, dtype=torch.float32)
    mask[idx.long()] = vals.float()
    return mask


def steer_sparse(sparse: torch.Tensor, mask: torch.Tensor, alpha: float, *, clamp_min_zero: bool = True) -> torch.Tensor:
    """Steer a sparse code along the edit direction.

    ``alpha > 0`` amplifies the concept, ``alpha < 0`` suppresses it. Suppression
    clamps at zero to respect ReLU-style non-negative SAE codes.
    """
    mask = mask.to(device=sparse.device, dtype=sparse.dtype)
    new_sparse = sparse + alpha * mask
    if clamp_min_zero and alpha < 0:
        new_sparse = new_sparse.clamp_min(0.0)
    return new_sparse


def decode_normalized(model, sparse: torch.Tensor) -> torch.Tensor:
    """Decode a sparse code to the dense space and L2-normalize (retrieval geometry)."""
    dense = model.sae_decoder(sparse)
    return F.normalize(dense, dim=-1)
