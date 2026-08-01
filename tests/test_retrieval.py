"""Tests for cosine nearest-neighbour retrieval."""
import torch

from steerable_retrieval.steer.retrieval import topk_cosine_neighbors


def test_topk_orders_and_excludes():
    gallery = torch.eye(6)
    query = gallery[2].clone()
    idx, scores = topk_cosine_neighbors(query, gallery, k=3)
    assert idx[0].item() == 2                     # nearest is itself
    assert scores[0] >= scores[-1]                # sorted descending

    idx2, _ = topk_cosine_neighbors(query, gallery, k=3, exclude_idx=2)
    assert 2 not in idx2.tolist()


def test_query_shape_agnostic():
    gallery = torch.randn(20, 8)
    q1, _ = topk_cosine_neighbors(gallery[0], gallery, k=5)
    q2, _ = topk_cosine_neighbors(gallery[0].reshape(1, -1), gallery, k=5)
    assert q1.tolist() == q2.tolist()


def test_k_capped_to_available():
    gallery = torch.randn(4, 3)
    idx, _ = topk_cosine_neighbors(gallery[0], gallery, k=100, exclude_idx=0)
    assert idx.shape[0] == 3
