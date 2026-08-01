"""Sparse steerable retrieval -- public API.

Open-vocabulary concept control for dense music retrieval via sparse inversion in a
trained SAE. See :class:`~steerable_retrieval.steer.slider.Slider` for the main entry point.
"""

from steerable_retrieval.steer.inversion import (
    InversionResult,
    MahalanobisPrior,
    fit_mahalanobis_prior,
    invert_concept,
    load_default_prior,
    mahalanobis_distance,
)
from steerable_retrieval.steer.retrieval import topk_cosine_neighbors
from steerable_retrieval.steer.steering import build_edit_mask, idf_weights, steer_sparse
from steerable_retrieval.steer.slider import Slider

__all__ = [
    "Slider",
    "InversionResult",
    "MahalanobisPrior",
    "fit_mahalanobis_prior",
    "load_default_prior",
    "invert_concept",
    "mahalanobis_distance",
    "build_edit_mask",
    "idf_weights",
    "steer_sparse",
    "topk_cosine_neighbors",
]
