"""
CKNNACallback — Centered Kernel Nearest-Neighbor Alignment between
SAE activations and input embeddings.

Logs CKNNA scores for four cross-space pairs:
  - text activations  <-> text embeddings
  - audio activations <-> audio embeddings
  - text activations  <-> audio embeddings
  - audio activations <-> text embeddings

Reads from pl_module.{val,test}_{activations,embeddings} populated by
LightningSAE._eval_step.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

from steerable_retrieval.callbacks.utils import BaseCallback
from steerable_retrieval.callbacks.energy import _get_dataset_name

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HSIC helpers (self-contained, no external deps)
# ---------------------------------------------------------------------------

def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al., JMLR 2012, Eq. 5)."""
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    hsic = (
        (K_tilde * L_tilde.T).sum()
        + K_tilde.sum() * L_tilde.sum() / ((m - 1) * (m - 2))
        - 2 * (K_tilde @ L_tilde).sum() / (m - 2)
    )
    return hsic / (m * (m - 3))


def _hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Biased HSIC (original CKA)."""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1.0 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


# ---------------------------------------------------------------------------
# CKNNA metric
# ---------------------------------------------------------------------------

def cknna(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    topk: int,
    distance_agnostic: bool = False,
    unbiased: bool = True,
) -> float:
    """
    Centered Kernel Nearest-Neighbor Alignment.

    Args:
        feats_A: [N, D_A] feature matrix.
        feats_B: [N, D_B] feature matrix.
        topk:    number of nearest neighbors (>= 2).
        distance_agnostic: if True, use binary neighbor overlap only.
        unbiased: if True, use unbiased HSIC and exclude self-similarities.

    Returns:
        CKNNA score (float).
    """
    n = feats_A.shape[0]
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    topk = min(topk, n - 1) if unbiased else min(topk, n)

    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    hsic_fn = _hsic_unbiased if unbiased else _hsic_biased

    def similarity(K_: torch.Tensor, L_: torch.Tensor, k: int) -> torch.Tensor:
        if unbiased:
            K_hat = K_.clone().fill_diagonal_(float("-inf"))
            L_hat = L_.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K_, L_

        _, topk_K_idx = torch.topk(K_hat, k, dim=1)
        _, topk_L_idx = torch.topk(L_hat, k, dim=1)

        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_idx, 1.0)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_idx, 1.0)
        mask = mask_K * mask_L

        if distance_agnostic:
            return (mask.sum()).float()
        return hsic_fn(mask * K_, mask * L_)

    sim_kl = similarity(K, L, topk)
    sim_kk = similarity(K, K, topk)
    sim_ll = similarity(L, L, topk)

    denom = (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()
    return sim_kl.item() / denom


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class CKNNACallback(BaseCallback):
    """
    Computes CKNNA between SAE activations and input embeddings at the end of
    each validation / test epoch.

    Pairs logged:
      - text_act   <-> text_emb
      - audio_act  <-> audio_emb
      - text_act   <-> audio_emb
      - audio_act  <-> text_emb
    """

    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        topk: int = 10,
        max_samples: int = 2048,
        distance_agnostic: bool = False,
        unbiased: bool = True,
        prefix: str = "CKNNA",
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.topk = topk
        self.max_samples = max_samples
        self.distance_agnostic = distance_agnostic
        self.unbiased = unbiased
        self.prefix = prefix

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.enable_on_validation:
            return
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        self._compute_and_log(trainer, pl_module, mode="val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.enable_on_test:
            return
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        self._compute_and_log(trainer, pl_module, mode="test")

    @staticmethod
    @torch.no_grad()
    def _compute(
        Za: Optional[torch.Tensor],
        Zt: Optional[torch.Tensor],
        Ea: Optional[torch.Tensor],
        Et: Optional[torch.Tensor],
        topk: int = 10,
        max_samples: int = 2048,
        distance_agnostic: bool = False,
        unbiased: bool = True,
    ) -> Dict[str, float]:
        """
        Pure computation — no trainer / pl_module / logging.

        Args:
            Za: audio activations [N_a, C] or None
            Zt: text activations  [N_t, C] or None
            Ea: audio embeddings  [N_a, D] or None
            Et: text embeddings   [N_t, D] or None
            topk: number of nearest neighbors for CKNNA
            max_samples: subsample limit
            distance_agnostic: binary neighbor overlap only
            unbiased: use unbiased HSIC

        Returns:
            dict mapping pair label -> CKNNA score (float).
        """
        pairs = [
            (Za, Ea, "audio_act__audio_emb"),
            (Zt, Et, "text_act__text_emb"),
            (Zt, Ea, "text_act__audio_emb"),
            (Za, Et, "audio_act__text_emb"),
        ]

        scores = {}
        for feats_A, feats_B, label in pairs:
            if feats_A is None or feats_B is None:
                continue

            n = min(feats_A.size(0), feats_B.size(0))
            fA, fB = feats_A[:n], feats_B[:n]

            if n > max_samples:
                idx = torch.randperm(n, device=fA.device)[:max_samples]
                fA, fB = fA[idx], fB[idx]

            try:
                scores[label] = cknna(
                    fA.float(), fB.float(),
                    topk=topk,
                    distance_agnostic=distance_agnostic,
                    unbiased=unbiased,
                )
            except Exception as e:
                log.warning(f"[CKNNACallback] Failed for {label}: {e}")

        return scores

    @torch.no_grad()
    def _compute_and_log(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        """Gather data from module, call _compute, then log."""
        from steerable_retrieval.callbacks.energy import _resolve_modality_tensors

        all_acts = getattr(pl_module, f"{mode}_activations", {})
        all_embs = getattr(pl_module, f"{mode}_embeddings", {})
        device = pl_module.device

        for dataloader_idx in all_acts:
            acts_dl = all_acts.get(dataloader_idx, {})
            embs_dl = all_embs.get(dataloader_idx, {})

            Za = _resolve_modality_tensors(acts_dl, "audio", device, trainer)
            Zt = _resolve_modality_tensors(acts_dl, "text", device, trainer)
            Ea = _resolve_modality_tensors(embs_dl, "audio", device, trainer)
            Et = _resolve_modality_tensors(embs_dl, "text", device, trainer)

            scores = self._compute(
                Za, Zt, Ea, Et,
                topk=self.topk,
                max_samples=self.max_samples,
                distance_agnostic=self.distance_agnostic,
                unbiased=self.unbiased,
            )

            dataset_name = _get_dataset_name(trainer, dataloader_idx, mode)
            log_prefix = f"{self.prefix}/{dataset_name}"
            for label, score in scores.items():
                pl_module.log(f"{log_prefix}/{label}", score, prog_bar=False, sync_dist=True)
