"""Cosine nearest-neighbour retrieval over a dense embedding corpus."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def topk_cosine_neighbors(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k: int,
    *,
    exclude_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(indices, scores)`` of the top-``k`` cosine neighbours of ``query``.

    Args:
        query: ``[d]`` or ``[1, d]`` dense query embedding.
        gallery: ``[N, d]`` corpus of dense embeddings.
        exclude_idx: optional gallery index to exclude (e.g. the seed track itself).
    """
    query = F.normalize(query.reshape(1, -1), dim=-1).squeeze(0)
    gallery = F.normalize(gallery, dim=-1)
    sims = gallery @ query
    if exclude_idx is not None:
        sims[exclude_idx] = -1e9
    n_avail = gallery.shape[0] - (1 if exclude_idx is not None else 0)
    k = min(int(k), int(n_avail))
    vals, idx = torch.topk(sims, k=k, largest=True, sorted=True)
    return idx.detach().cpu(), vals.detach().cpu()
