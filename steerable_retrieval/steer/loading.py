"""Loading trained SAEs for the steering API.

A :class:`SteerableSAE` is the minimal model surface the :class:`~steerable_retrieval.steer.slider.Slider`
needs: a trained ``sae_encoder`` + ``sae_decoder`` and a ``text_encoder`` mapping concept
strings to embeddings in the same joint space. This module resolves such a model from a
checkpoint (local path or HuggingFace repo).

NOTE: the released pretrained checkpoint (BatchTopK SAE on MuQ / music4all) is not published
yet. Until then, construct a model in-memory and pass it via ``Slider(..., model=model)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SteerableSAE(nn.Module):
    """Minimal SAE wrapper for inference-time steering (no training deps).

    Args:
        sae_encoder: a trained SAE encoder (``steerable_retrieval.models.sae.encoders``).
        sae_decoder: a trained SAE decoder (``steerable_retrieval.models.sae.decoders``).
        text_encoder: callable mapping ``list[str] -> [B, d]`` embeddings (e.g. the MuQ
            text tower), living in the same joint space as the SAE's audio inputs.
    """

    def __init__(self, sae_encoder, sae_decoder, text_encoder):
        super().__init__()
        self.sae_encoder = sae_encoder
        self.sae_decoder = sae_decoder
        self.text_encoder = text_encoder
        if not hasattr(self.sae_decoder, "b_dec"):
            self.sae_decoder.b_dec = self.sae_encoder.b_dec

    @torch.no_grad()
    def inference(self, x):
        """Encode dense features to a sparse code and back. Returns ``(x, z, xhat, pre)``."""
        pre, z = self.sae_encoder(x)
        xhat = self.sae_decoder(z)
        return x, z, xhat, pre


def _resolve_local_run(checkpoint: str, subfolder: Optional[str] = None):
    """Return (ckpt_path, config_path) for a checkpoint, downloading from the Hub if
    ``checkpoint`` is a ``org/repo`` id rather than a local path.

    A Lightning run stores its Hydra config at ``<run>/.hydra/config.yaml`` and its
    checkpoints under ``<run>/checkpoints/``. We use that config to rebuild the SAE
    modules before loading the (encoder/decoder-only) weights. For a Hub repo carrying
    several models, ``subfolder`` (e.g. ``"L0-20"``) selects which one.
    """
    import os

    if os.path.exists(checkpoint):
        ckpt_path = os.path.abspath(checkpoint)
        run_dir = os.path.dirname(os.path.dirname(ckpt_path))  # .../checkpoints/x.ckpt -> run
        cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if not os.path.exists(cfg_path):
            # allow a config.yaml sitting next to the checkpoint
            alt = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
            cfg_path = alt if os.path.exists(alt) else cfg_path
        return ckpt_path, cfg_path

    # Otherwise treat it as a HuggingFace repo id: expects last.ckpt + config.yaml
    # (optionally under `subfolder`).
    from huggingface_hub import hf_hub_download

    pre = f"{subfolder}/" if subfolder else ""
    ckpt_path = hf_hub_download(checkpoint, filename=f"{pre}last.ckpt")
    try:
        cfg_path = hf_hub_download(checkpoint, filename=f"{pre}config.yaml")
    except Exception:
        cfg_path = hf_hub_download(checkpoint, filename=f"{pre}.hydra/config.yaml")
    return ckpt_path, cfg_path


def load_steerable_sae(
    checkpoint: Optional[str],
    *,
    model_class: Optional[str] = None,
    device: str = "cpu",
    text_encoder=None,
    build_text_encoder: bool = True,
    config_path: Optional[str] = None,
    subfolder: Optional[str] = None,
) -> SteerableSAE:
    """Resolve a :class:`SteerableSAE` from a trained Lightning checkpoint.

    Args:
        checkpoint: local path to a ``.ckpt`` (its run's ``.hydra/config.yaml`` is used
            to rebuild the SAE modules), or a HuggingFace ``org/repo`` id carrying
            ``last.ckpt`` + ``config.yaml``.
        device: where to place the model.
        text_encoder: a ready callable ``list[str] -> [B, d]``. If ``None`` and
            ``build_text_encoder`` is True, the text tower from the run config (e.g.
            MuQ-MuLan) is instantiated; if False, ``text_encoder`` stays ``None`` (useful
            for steering/retrieval that never embeds new text).
        config_path: override the run config path explicitly.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    ckpt_path, resolved_cfg = _resolve_local_run(checkpoint, subfolder=subfolder)
    cfg = OmegaConf.load(config_path or resolved_cfg)

    enc = instantiate(cfg.model.sae_encoder, device=device)
    dec = instantiate(cfg.model.sae_decoder, device=device)
    if not hasattr(dec, "b_dec"):
        dec.b_dec = enc.b_dec

    # Lightning checkpoints from our training runs can contain OmegaConf metadata
    # alongside tensors. PyTorch 2.6 defaults torch.load(weights_only=True), which
    # rejects that metadata; this loader is for trusted project checkpoints.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("state_dict", state)
    enc_sd = {k[len("sae_encoder."):]: v for k, v in sd.items() if k.startswith("sae_encoder.")}
    dec_sd = {k[len("sae_decoder."):]: v for k, v in sd.items() if k.startswith("sae_decoder.")}
    if not enc_sd or not dec_sd:
        raise ValueError(
            f"No sae_encoder/sae_decoder weights found in {ckpt_path}. "
            f"Available prefixes: {sorted({k.split('.')[0] for k in sd})}"
        )
    enc.load_state_dict(enc_sd, strict=False)
    dec.load_state_dict(dec_sd, strict=False)

    if text_encoder is None and build_text_encoder:
        text_encoder = instantiate(cfg.model.text_encoder, device=device)

    model = SteerableSAE(enc, dec, text_encoder)
    model.to(device).eval()
    return model
