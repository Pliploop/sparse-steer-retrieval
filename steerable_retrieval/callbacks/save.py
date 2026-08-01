"""
SaveActivationsCallback — saves validation embeddings, preactivations,
activations, and reconstructions to per-tensor .pt files at the end of each
validation epoch.

Reads embeddings and activations stored on the LightningModule by _eval_step,
then re-runs the SAE forward pass to capture preactivations and reconstructions
(which are not stored on the module to save memory during normal training).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

from steerable_retrieval.callbacks.utils import BaseCallback
from steerable_retrieval.callbacks.energy import _get_dataset_name, gather_tensor_if_distributed, _cat_or_none

log = logging.getLogger(__name__)


class SaveActivationsCallback(BaseCallback):
    """
    At the end of every validation epoch, saves separate .pt files containing
    embeddings, preactivations, activations, and reconstructions for every
    validation dataloader / modality.

    File structure::

        <save_dir>/activations/<dataset_name>/<modality>/<tensor_name>/tensors.pt

    Example::

        activations/dataset_name/audio/embeddings/tensors.pt
        activations/dataset_name/audio/preactivations/tensors.pt
        activations/dataset_name/audio/activations/tensors.pt
        activations/dataset_name/audio/reconstructions/tensors.pt
    """

    ROOT_DIRNAME = "activations"

    def __init__(
        self,
        save_dir: str,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not (self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)):
            return
        self._save(trainer, pl_module)

    @torch.no_grad()
    def _save(self, trainer: Trainer, pl_module: LightningModule):
        """Build tensors from module-stored embeddings/activations and save to disk."""
        device = pl_module.device
        all_acts = getattr(pl_module, "val_activations", {})
        all_embs = getattr(pl_module, "val_embeddings", {})

        for dataloader_idx in all_acts:
            dataset_name = _get_dataset_name(trainer, dataloader_idx, "val")

            acts_dl = all_acts.get(dataloader_idx, {})
            embs_dl = all_embs.get(dataloader_idx, {})

            for modality in ("audio", "text"):
                act_chunks = acts_dl.get(modality, [])
                emb_chunks = embs_dl.get(modality, [])
                if not act_chunks:
                    continue

                # Concatenate and gather
                activations = _cat_or_none(act_chunks, device=device)
                embeddings = _cat_or_none(emb_chunks, device=device) if emb_chunks else None

                activations = gather_tensor_if_distributed(activations, trainer)
                if embeddings is not None:
                    embeddings = gather_tensor_if_distributed(embeddings, trainer)

                # Re-run SAE forward on embeddings to get preactivations & reconstructions
                preactivations = None
                reconstructions = None
                if embeddings is not None:
                    # Process in chunks to avoid OOM
                    chunk_size = 512
                    pre_chunks, rec_chunks = [], []
                    for i in range(0, embeddings.size(0), chunk_size):
                        emb_chunk = embeddings[i : i + chunk_size].to(device)
                        _, z, xhat, pre = pl_module(emb_chunk)
                        pre_chunks.append(pre.cpu() if pre is not None else torch.zeros_like(z).cpu())
                        rec_chunks.append(xhat.cpu())
                    preactivations = torch.cat(pre_chunks, dim=0)
                    reconstructions = torch.cat(rec_chunks, dim=0)

                tensors = {
                    "embeddings": embeddings.cpu() if embeddings is not None else None,
                    "preactivations": preactivations,
                    "activations": activations.cpu(),
                    "reconstructions": reconstructions,
                }
                if trainer.is_global_zero:
                    self._save_tensors(dataset_name=dataset_name, modality=modality, tensors=tensors)

        # Only rank-0 writes to disk
        if trainer.is_global_zero:
            log.info(f"[SaveActivationsCallback] Saved validation activations to {self.save_dir}")

    def _save_tensors(
        self,
        dataset_name: str,
        modality: str,
        tensors: Dict[str, Optional[torch.Tensor]],
    ) -> None:
        base_dir = os.path.join(self.save_dir, self.ROOT_DIRNAME, dataset_name, modality)
        for tensor_name, tensor_value in tensors.items():
            if tensor_value is None:
                continue
            tensor_dir = os.path.join(base_dir, tensor_name)
            os.makedirs(tensor_dir, exist_ok=True)
            tensor_path = os.path.join(tensor_dir, "tensors.pt")
            torch.save(tensor_value, tensor_path)
