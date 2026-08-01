"""
Callbacks implementing Energy / Modality / Bridge / Cross-run Stability metrics
for LightningSAE (spamr) based on the PCA/USAE paper (2504.11695v4).
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

from steerable_retrieval.callbacks.utils import BaseCallback

import plotly.graph_objects as go

log = logging.getLogger(__name__)

# Bar plot styling: normalized neuron index (0-1), fixed width, black outline
BAR_WIDTH = 0.005
BAR_MARKER_LINE = dict(width=1, color="black")


def _normalized_neuron_x(n: int):
    """Return x-axis values in [0, 1] for n neurons."""
    if n <= 1:
        return [0.5] if n == 1 else []
    return [i / (n - 1) for i in range(n)]


def _get_wandb_logger(trainer: Trainer):
    """Get wandb logger from trainer if available."""
    try:
        from lightning.pytorch.loggers import WandbLogger
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
    except ImportError:
        pass
    return None


def _get_dataset_name(trainer: Trainer, dataloader_idx: int, mode: str) -> str:
    """Get dataset name for a dataloader from datamodule.names, or fallback to dataloader_{idx}."""
    dm = getattr(trainer, "datamodule", None)
    if dm is not None and hasattr(dm, "names"):
        names = dm.names
        if isinstance(names, dict) and mode in names:
            name = names[mode].get(dataloader_idx)
            if name is not None:
                return str(name)
    return f"dataloader_{dataloader_idx}"


def _log_plotly_to_wandb(trainer: Trainer, key: str, fig):
    """Log a plotly figure to wandb if available."""
    wandb_logger = _get_wandb_logger(trainer)
    if wandb_logger is None:
        return
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({key: fig}, step=trainer.global_step)
    except ImportError:
        pass


def gather_tensor_if_distributed(tensor: torch.Tensor, trainer: Trainer) -> torch.Tensor:
    """
    Gather tensor from all processes if distributed training is enabled.
    Uses PyTorch Lightning's strategy for distributed operations.
    """
    if trainer.world_size <= 1:
        return tensor

    try:
        if hasattr(trainer.strategy, "all_gather"):
            gathered = trainer.strategy.all_gather(tensor, sync_grads=False)
            if isinstance(gathered, (list, tuple)):
                gathered = torch.cat(gathered, dim=0)
            elif isinstance(gathered, torch.Tensor):
                if gathered.dim() > tensor.dim():
                    gathered = gathered.view(-1, *gathered.shape[2:])
            return gathered
    except (AttributeError, NotImplementedError):
        pass

    import torch.distributed as dist
    if not dist.is_initialized():
        return tensor

    device = tensor.device

    local_size = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(trainer.world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [s.item() for s in sizes]
    max_size = max(sizes)

    if tensor.shape[0] < max_size:
        padding_shape = list(tensor.shape)
        padding_shape[0] = max_size - tensor.shape[0]
        padding = torch.zeros(padding_shape, device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(trainer.world_size)]
    dist.all_gather(gathered_tensors, tensor)

    gathered_list = []
    for i, gathered_tensor in enumerate(gathered_tensors):
        gathered_list.append(gathered_tensor[:sizes[i]])

    return torch.cat(gathered_list, dim=0)


def get_dictionary_from_lightningsae(pl_module: LightningModule) -> torch.Tensor:
    """
    Returns dictionary D as [C, d] where each row is a concept atom in embedding space.

    The SAE decoder has W_dec of shape [dict_size, act_size] = [C, d].
    """
    if not hasattr(pl_module, "sae_decoder"):
        raise AttributeError("Expected pl_module.sae_decoder to exist (LightningSAE).")

    W = getattr(pl_module.sae_decoder, "W_dec", None)
    if W is None or not torch.is_tensor(W):
        raise AttributeError("Expected pl_module.sae_decoder.W_dec to be a Tensor/Parameter.")

    return W.contiguous()


def load_state_dict_any(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    """
    With torch_s3_connector installed, torch.load("s3://...") should work.

    Supports:
      - raw state_dict
      - Lightning checkpoint dict containing 'state_dict'
    """
    if str(path).startswith("s3://"):
        from urllib.parse import urlparse

        import boto3

        parsed = urlparse(str(path))
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 checkpoint URI: {path}")

        cache_root = os.environ.get("SPAMR_CHECKPOINT_CACHE_DIR", os.path.expanduser("~/.cache/steerable_retrieval/checkpoints"))
        cache_path = os.path.join(cache_root, bucket, key)
        force_refresh = str(os.environ.get("SPAMR_CHECKPOINT_CACHE_REFRESH", "0")).lower() in {"1", "true", "yes"}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if force_refresh or not os.path.exists(cache_path):
            boto3.client("s3").download_file(bucket, key, cache_path)
            log.info(f"Downloaded checkpoint to local cache: {cache_path}")
        else:
            log.info(f"Using cached checkpoint: {cache_path}")

        # Checkpoints in this project may include OmegaConf objects in metadata.
        # PyTorch>=2.6 defaults to weights_only=True, which rejects those objects.
        obj = torch.load(cache_path, map_location=map_location, weights_only=False)
    else:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint object at {path}: {type(obj)}")


def _safe_name(path: str) -> str:
    base = path.rstrip("/").split("/")[-1]
    base = base.replace(".", "_").replace("-", "_").replace("=", "_")
    if len(base) > 48:
        base = base[:48]
    return base


def _cat_or_none(xs: List[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if not xs:
        return None
    return torch.cat([x.to(device) for x in xs], dim=0)


def _resolve_modality_tensors(
    store: Dict,
    modality: str,
    device: torch.device,
    trainer: Optional["Trainer"] = None,
) -> Optional[torch.Tensor]:
    """
    Resolve a modality's data from a per-dataloader storage dict.
    Handles both list-of-tensors and already-concatenated tensor formats.
    Optionally gathers across ranks when trainer is provided.
    """
    data = store.get(modality)
    if data is None:
        return None
    if isinstance(data, list):
        t = _cat_or_none(data, device=device)
    elif isinstance(data, torch.Tensor):
        t = data.to(device)
    else:
        return None
    if t is not None and trainer is not None:
        t = gather_tensor_if_distributed(t, trainer)
    return t


# -----------------------------------------------------------------------------
# EnergyCallback
# -----------------------------------------------------------------------------
class EnergyCallback(BaseCallback):
    """
    Energy_i = E[z_i] computed from logged SAE activations.

    Logs:
      - {mode}/energy_sum
      - {mode}/energy_mean
      - {mode}/energy_topK_frac   (fraction of total energy contained in top-K concepts)

    Also stores:
      pl_module._last_energy = {'combined': E, 'audio': Ea or None, 'text': Et or None}
    """

    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        topk: int = 512,
        prefix: str = "energy",
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.topk = topk
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
        self._compute_and_log(trainer, pl_module, mode="test")

    @staticmethod
    @torch.no_grad()
    def _compute(
        Za: Optional[torch.Tensor],
        Zt: Optional[torch.Tensor],
        topk: int = 512,
    ) -> Optional[Dict]:
        """
        Pure computation — no trainer / pl_module / logging.

        Args:
            Za: audio activations [N_a, C] or None
            Zt: text activations  [N_t, C] or None
            topk: number of top concepts for energy fraction

        Returns:
            dict with keys: E, Ea, Et, total, frac_topk, k, Za, Zt
            or None if both inputs are None.
        """
        if Za is None and Zt is None:
            return None

        Ea = Za.mean(dim=0) if Za is not None else None
        Et = Zt.mean(dim=0) if Zt is not None else None

        if Ea is not None and Et is not None:
            E = 0.5 * (Ea + Et)
        else:
            E = Ea if Ea is not None else Et

        total = E.sum().clamp_min(1e-12)
        k = min(topk, E.numel())
        frac_topk = (torch.topk(E, k=k).values.sum() / total).item()

        return dict(E=E, Ea=Ea, Et=Et, total=total, frac_topk=frac_topk, k=k, Za=Za, Zt=Zt)

    @torch.no_grad()
    def _compute_and_log(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        """Gather data from module, call _compute, then log."""
        all_acts = pl_module.val_activations if mode == "val" else pl_module.test_activations
        device = pl_module.device

        for dataloader_idx, acts in all_acts.items():
            Za = _resolve_modality_tensors(acts, "audio", device, trainer)
            Zt = _resolve_modality_tensors(acts, "text", device, trainer)

            result = self._compute(Za, Zt, topk=self.topk)
            if result is None:
                log.warning(f"[EnergyCallback] No activations found for mode={mode}, dataloader_idx={dataloader_idx}.")
                continue

            E, Ea, Et = result["E"], result["Ea"], result["Et"]
            dataset_name = _get_dataset_name(trainer, dataloader_idx, mode)
            log_prefix = f"Energy/{dataset_name}"
            pl_module.log(f"{log_prefix}/sum", result["total"].item(), sync_dist=True)
            pl_module.log(f"{log_prefix}/mean", E.mean().item(), sync_dist=True)
            pl_module.log(f"{log_prefix}/top{result['k']}_frac", result["frac_topk"], prog_bar=True, sync_dist=True)

            if dataloader_idx == 0 or len(all_acts) == 1:
                pl_module._last_energy = {
                    "combined": E.detach(),
                    "audio": Ea.detach() if Ea is not None else None,
                    "text": Et.detach() if Et is not None else None,
                }

            if Za is not None and Zt is not None:
                self._log_energy_distribution_combined(trainer, Za, Zt, mode, dataset_name)
            for activations, name in zip([Za, Zt], ["audio", "text"]):
                if activations is not None:
                    self._log_energy_distribution(trainer, activations, name, mode, dataset_name)

            self._log_cumulative_energy_plot(trainer, pl_module, E, mode, dataset_name)

    def _log_energy_distribution_combined(
        self,
        trainer: Trainer,
        Za: torch.Tensor,
        Zt: torch.Tensor,
        mode: str,
        dataset_name: str,
    ):
        """Stacked bar: mean activation per neuron from audio (blue) and text (red)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.debug("plotly not available, skipping energy distribution combined plot")
            return

        mean_a = Za.detach().cpu().mean(dim=0)
        mean_t = Zt.detach().cpu().mean(dim=0)
        n_concepts = mean_a.shape[0]
        x_norm = _normalized_neuron_x(n_concepts)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_norm,
            y=mean_a.numpy().tolist(),
            name="audio",
            width=BAR_WIDTH,
            marker=dict(color="blue", line=BAR_MARKER_LINE),
            hovertemplate="Neuron: %{x:.4f}<br>Audio: %{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=x_norm,
            y=mean_t.numpy().tolist(),
            name="text",
            width=BAR_WIDTH,
            marker=dict(color="red", line=BAR_MARKER_LINE),
            hovertemplate="Neuron: %{x:.4f}<br>Text: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            barmode="stack",
            title=f"Energy Distribution — combined ({mode})",
            xaxis_title="Neuron Index (normalized 0–1)",
            yaxis_title="Mean Activation",
            template="plotly_white",
        )
        _log_plotly_to_wandb(trainer, f"Energy Distribution/{dataset_name}/energy_distribution_combined", fig)

    def _log_energy_distribution(
        self,
        trainer: Trainer,
        activations: torch.Tensor,
        name: str,
        mode: str,
        dataset_name: str,
    ):
        """Plot average activation per neuron (bar height = mean, error bars = std from raw data)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.debug("plotly not available, skipping energy distribution plot")
            return

        acts = activations.detach().cpu()
        mean_per_neuron = acts.mean(dim=0)
        std_per_neuron = acts.std(dim=0)
        n_concepts = mean_per_neuron.shape[0]

        logging.info(f"dataset_name: {dataset_name}, modality: {name}, mean_per_neuron: {mean_per_neuron.shape}, std_per_neuron: {std_per_neuron.shape}, n_concepts: {n_concepts}")
        
        # check if nans in activations

        if torch.isnan(mean_per_neuron).any() or torch.isnan(std_per_neuron).any():
            log.warning(f"NaNs found in activations for dataset_name: {dataset_name}, modality: {name}")
            
        logging.info(f"neurons with nans: {torch.isnan(mean_per_neuron).nonzero()}, {torch.isnan(std_per_neuron).nonzero()}")
        logging.info(f'mean_per_neuron: {mean_per_neuron}, std_per_neuron: {std_per_neuron}')


        # Modality color: blue for audio, red for text
        color = "blue" if name == "audio" else "red"
        x_norm = _normalized_neuron_x(n_concepts)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_norm,
            y=mean_per_neuron.numpy().tolist(),
            name=name,
            width=BAR_WIDTH,
            marker=dict(color=color, line=BAR_MARKER_LINE),
            hovertemplate="Neuron: %{x:.4f}<br>Mean: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Energy Distribution — {name} ({mode})",
            xaxis_title="Neuron Index (normalized 0–1)",
            yaxis_title="Mean Activation",
            template="plotly_white",
        )
        _log_plotly_to_wandb(trainer, f"Energy Distribution/{dataset_name}/energy_distribution_{name}", fig)

    def _log_cumulative_energy_plot(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        E: torch.Tensor,
        mode: str,
        dataset_name: str,
        log_scale: bool = True,
    ):
        """
        Log cumulative energy plot: concepts sorted by energy (descending),
        x-axis = concept rank, y-axis = cumulative normalized energy.
        If log_scale is True (default), x-axis is log-scaled.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.debug("plotly not available, skipping cumulative energy plot")
            return

        # Sort energies descending
        E_sorted, _ = torch.sort(E, descending=True)
        E_sorted = E_sorted.detach().cpu()
        total_energy = E_sorted.sum().clamp_min(1e-12)

        # Compute cumulative normalized energy
        cumulative = torch.cumsum(E_sorted, dim=0) / total_energy
        cumulative = cumulative.numpy()

        n_concepts = len(cumulative)
        x = list(range(1, n_concepts + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=cumulative.tolist(),
            mode="lines",
            name="Cumulative Energy",
            line=dict(color="blue", width=2),
        ))
        fig.update_layout(
            title=f"Cumulative Normalized Energy ({mode})",
            xaxis_title="Number of Concepts (ranked by energy)",
            yaxis_title="Cumulative Energy Fraction",
            yaxis_range=[0, 1.05],
            template="plotly_white",
        )

        if log_scale:
            fig.update_xaxes(type="log")

        plot_key = f"Energy Distribution/{dataset_name}/cumulative_plot"
        _log_plotly_to_wandb(trainer, plot_key, fig)


# -----------------------------------------------------------------------------
# ModalityScoreCallback
# -----------------------------------------------------------------------------
class ModalityScoreCallback(BaseCallback):
    """
    ModalityScore_i = E_audio[z_i] / (E_audio[z_i] + E_text[z_i])

    Logs summary stats:
      - mean modality score
      - fraction mostly-audio (>0.9)
      - fraction mostly-text (<0.1)

    Stores:
      pl_module._last_modality_score = score
    """

    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        eps: float = 1e-5,
        prefix: str = "modality",
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.eps = eps
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
        Za: torch.Tensor,
        Zt: torch.Tensor,
        eps: float = 1e-5,
    ) -> Dict:
        """
        Pure computation — no trainer / pl_module / logging.

        Args:
            Za: audio activations [N_a, C]
            Zt: text activations  [N_t, C]
            eps: epsilon for numerical stability

        Returns:
            dict with keys: score, E, Ea, Et, mean, frac_audio, frac_text
        """
        Ea = Za.mean(dim=0)
        Et = Zt.mean(dim=0)
        score = Ea / (Ea + Et + eps)
        E = 0.5 * (Ea + Et)
        return dict(
            score=score,
            E=E,
            Ea=Ea,
            Et=Et,
            mean=score.mean().item(),
            frac_audio=((score > 0.9).float().mean().item()),
            frac_text=((score < 0.1).float().mean().item()),
        )

    @torch.no_grad()
    def _compute_and_log(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        """Gather data from module, call _compute, then log."""
        all_acts = pl_module.val_activations if mode == "val" else pl_module.test_activations
        device = pl_module.device

        for dataloader_idx, acts in all_acts.items():
            Za = _resolve_modality_tensors(acts, "audio", device, trainer)
            Zt = _resolve_modality_tensors(acts, "text", device, trainer)

            if Za is None or Zt is None:
                log.warning(f"[ModalityScoreCallback] Missing modality activations for mode={mode}, dataloader_idx={dataloader_idx}.")
                continue

            result = self._compute(Za, Zt, eps=self.eps)
            score, E = result["score"], result["E"]

            dataset_name = _get_dataset_name(trainer, dataloader_idx, mode)
            log_prefix = f"Modality Score/{dataset_name}"
            pl_module.log(f"{log_prefix}/mean", result["mean"], prog_bar=True, sync_dist=True)
            pl_module.log(f"{log_prefix}/frac_audio_gt0.9", result["frac_audio"], sync_dist=True)
            pl_module.log(f"{log_prefix}/frac_text_lt0.1", result["frac_text"], sync_dist=True)

            if dataloader_idx == 0 or len(all_acts) == 1:
                pl_module._last_modality_score = score.detach()

            self._log_modality_histograms(trainer, score, E, mode, dataset_name)
            self._log_modality_score_distribution(trainer, score, mode, dataset_name)

    @torch.no_grad()
    def _log_modality_score_distribution(
        self,
        trainer: Trainer,
        score: torch.Tensor,
        mode: str,
        dataset_name: str,
    ):
        """Plot modality score per neuron index using plotly."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.debug("plotly not available, skipping modality score distribution plot")
            return

        score = score.detach().cpu()
        n = score.shape[0]
        x_norm = _normalized_neuron_x(n)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_norm,
            y=score.numpy().tolist(),
            name="Modality Score",
            width=BAR_WIDTH,
            marker=dict(color="steelblue", line=BAR_MARKER_LINE),
        ))
        fig.update_layout(
            title=f"Modality Score Distribution ({mode})",
            xaxis_title="Neuron Index (normalized 0–1)",
            yaxis_title="Modality Score",
            template="plotly_white",
        )
        _log_plotly_to_wandb(trainer, f"Modality Score/{dataset_name}/modality_score_distribution", fig)

    def _log_modality_histograms(
        self,
        trainer: Trainer,
        score: torch.Tensor,
        E: torch.Tensor,
        mode: str,
        dataset_name: str,
    ):
        """
        Log two histograms of modality scores per concept:
        - Unweighted: count of concepts per bin (normalized to proportion).
        - Energy-weighted: energy in each bin (normalized by total energy).
        """
        try:
            import numpy as np
            import plotly.graph_objects as go
        except ImportError:
            log.debug("numpy/plotly not available, skipping modality histograms")
            return

        score_np = score.detach().cpu().float().numpy()
        E_np = E.detach().cpu().float().numpy()
        bins = np.linspace(0.0, 1.0, 51)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Unweighted: count per bin, normalize to proportion
        counts, _ = np.histogram(score_np, bins=bins)
        total = max(counts.sum(), 1e-12)
        density_unweighted = (counts / total).tolist()

        # Energy-weighted: sum of energy per bin, normalize by total energy
        counts_w, _ = np.histogram(score_np, bins=bins, weights=E_np)
        total_E = max(E_np.sum(), 1e-12)
        density_weighted = (counts_w / total_E).tolist()

        layout_base = dict(
            xaxis_title="Modality Score (0=text, 1=audio)",
            yaxis_title="Proportion",
            template="plotly_white",
            bargap=0.1,
        )

        fig_unweighted = go.Figure()
        fig_unweighted.add_trace(go.Bar(
            x=bin_centers.tolist(),
            y=density_unweighted,
            marker=dict(color="steelblue", line=BAR_MARKER_LINE),
            width=0.005,
        ))
        fig_unweighted.update_layout(
            title=f"Modality Score Distribution — Unweighted ({mode})",
            **layout_base,
        )
        fig_unweighted.update_yaxes(rangemode="tozero")
        _log_plotly_to_wandb(
            trainer,
            f"Modality Score/{dataset_name}/distribution_unweighted",
            fig_unweighted,
        )

        fig_weighted = go.Figure()
        fig_weighted.add_trace(go.Bar(
            x=bin_centers.tolist(),
            y=density_weighted,
            marker=dict(color="coral", line=BAR_MARKER_LINE),
            width=0.005,
        ))
        fig_weighted.update_layout(
            title=f"Modality Score Distribution — Energy-Weighted ({mode})",
            **layout_base,
        )
        fig_weighted.update_yaxes(rangemode="tozero")
        _log_plotly_to_wandb(
            trainer,
            f"Modality Score/{dataset_name}/distribution_energy_weighted",
            fig_weighted,
        )


# -----------------------------------------------------------------------------
# BridgeScoreCallback
# -----------------------------------------------------------------------------
class BridgeScoreCallback(BaseCallback):
    """
    Bridge matrix:
      B = E[z_a^T z_t] ⊙ (D D^T)
    where:
      - E[z_a^T z_t] is estimated as (Za^T Zt)/N for paired samples
      - D D^T uses cosine similarity between dictionary atoms (rows of D)

    Logs only matrix summaries (avoid logging full B):
      - abs mean
      - abs max
      - abs topK mean

    Stores:
      pl_module._last_bridge = B
    """

    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        prefix: str = "bridge",
        topk_edges: int = 200,
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.prefix = prefix
        self.topk_edges = topk_edges

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
        Za: torch.Tensor,
        Zt: torch.Tensor,
        W_dec: torch.Tensor,
        topk_edges: int = 200,
    ) -> Dict:
        """
        Pure computation — no trainer / pl_module / logging.

        Args:
            Za: audio activations [N, C]
            Zt: text activations  [N, C]  (must be same N)
            W_dec: decoder dictionary [C, d]
            topk_edges: number of top bridge edges

        Returns:
            dict with keys: B, abs_mean, abs_max, abs_topk_mean, k
        """
        if Za.shape[0] != Zt.shape[0]:
            raise ValueError(f"BridgeScore requires paired N. Got Za={Za.shape}, Zt={Zt.shape}")

        N = Za.shape[0]
        coact = (Za.t() @ Zt) / max(N, 1)

        Dn = F.normalize(W_dec, dim=-1)
        align = Dn @ Dn.t()
        B = coact * align

        absB = B.abs()
        k = min(topk_edges, absB.numel())
        top_vals = torch.topk(absB.cpu().flatten(), k=k).values

        return dict(
            align=align,
            coact=coact,
            B=B,
            abs_mean=absB.mean().item(),
            abs_max=absB.max().item(),
            abs_topk_mean=top_vals.mean().item(),
            k=k,
        )

    @torch.no_grad()
    def _compute_and_log(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        """Gather data from module, call _compute, then log."""
        all_acts = pl_module.val_activations if mode == "val" else pl_module.test_activations
        device = pl_module.device

        bridges = {}

        for dataloader_idx, acts in all_acts.items():
            Za = _resolve_modality_tensors(acts, "audio", device, trainer)
            Zt = _resolve_modality_tensors(acts, "text", device, trainer)

            if Za is None or Zt is None:
                log.warning(f"[BridgeScoreCallback] Missing modality activations for mode={mode}, dataloader_idx={dataloader_idx}.")
                continue

            W_dec = get_dictionary_from_lightningsae(pl_module).to(device)
            result = self._compute(Za, Zt, W_dec, topk_edges=self.topk_edges)

            dataset_name = _get_dataset_name(trainer, dataloader_idx, mode)
            bridges[dataset_name] = {
                "align": result["align"],
                "coact": result["coact"],
                "B": result["B"],
            }
            log_prefix = f"Bridge/{dataset_name}"
            pl_module.log(f"{log_prefix}/abs_mean", result["abs_mean"], prog_bar=True, sync_dist=True)
            pl_module.log(f"{log_prefix}/abs_max", result["abs_max"], sync_dist=True)
            pl_module.log(f"{log_prefix}/abs_top{result['k']}_mean", result["abs_topk_mean"], sync_dist=True)

            if dataloader_idx == 0 or len(all_acts) == 1:
                pl_module._last_bridge = result["B"].detach()

        # Preferred structure: per-dataset bridge tensors.
        # Example: pl_module.bridges[dataset_name]["align" | "coact" | "B"]
        pl_module.bridges = bridges


# -----------------------------------------------------------------------------
# ModalityClassifierCallback
# -----------------------------------------------------------------------------
class ModalityClassifierCallback(BaseCallback):
    """
    Per-concept modality classification accuracy.

    For each concept, trains a logistic regression classifier to distinguish
    audio vs text embeddings based on that concept's activation value.

    Logs:
      - Mean accuracy across all concepts
      - Histogram of per-concept accuracies (unweighted)
      - Histogram of per-concept accuracies weighted by energy

    This helps identify which concepts are most discriminative for modality.
    """

    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        prefix: str = "modality_classifier",
        max_samples: int = 10000,
        test_fraction: float = 0.2,
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.prefix = prefix
        self.max_samples = max_samples
        self.test_fraction = test_fraction

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
        Za: torch.Tensor,
        Zt: torch.Tensor,
        max_samples: int = 10000,
        test_fraction: float = 0.2,
    ) -> Optional[Dict]:
        """
        Pure computation — no trainer / pl_module / logging.

        Args:
            Za: audio activations [N_a, C]
            Zt: text activations  [N_t, C]
            max_samples: subsample limit per modality
            test_fraction: held-out fraction for classifier

        Returns:
            dict with keys: accuracies (np.ndarray [C]), E_np (np.ndarray [C]),
            mean_acc, energy_weighted_mean_acc
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
        except ImportError:
            log.warning("[ModalityClassifierCallback] sklearn not available, skipping.")
            return None

        if Za.shape[0] > max_samples:
            Za = Za[torch.randperm(Za.shape[0])[:max_samples]]
        if Zt.shape[0] > max_samples:
            Zt = Zt[torch.randperm(Zt.shape[0])[:max_samples]]

        Za_np = Za.cpu().float().numpy()
        Zt_np = Zt.cpu().float().numpy()
        n_concepts = Za_np.shape[1]

        Ea = Za.mean(dim=0)
        Et = Zt.mean(dim=0)
        E = 0.5 * (Ea + Et)
        E_np = E.cpu().float().numpy()

        y_audio = np.ones(Za_np.shape[0], dtype=np.int32)
        y_text = np.zeros(Zt_np.shape[0], dtype=np.int32)

        accuracies = []
        for c in range(n_concepts):
            X_c = np.concatenate([Za_np[:, c : c + 1], Zt_np[:, c : c + 1]], axis=0)
            y_c = np.concatenate([y_audio, y_text], axis=0)
            if X_c.std() < 1e-9:
                accuracies.append(0.5)
                continue
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_c, y_c, test_size=test_fraction, stratify=y_c, random_state=42
                )
                clf = LogisticRegression(max_iter=200, solver="lbfgs")
                clf.fit(X_train, y_train)
                accuracies.append(clf.score(X_test, y_test))
            except Exception:
                accuracies.append(0.5)

        accuracies = np.array(accuracies)
        mean_acc = float(accuracies.mean())
        ew_mean_acc = float((accuracies * E_np).sum() / max(E_np.sum(), 1e-12))

        return dict(
            accuracies=accuracies,
            E_np=E_np,
            mean_acc=mean_acc,
            energy_weighted_mean_acc=ew_mean_acc,
        )

    @torch.no_grad()
    def _compute_and_log(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        """Gather data from module, call _compute, then log."""
        all_acts = pl_module.val_activations if mode == "val" else pl_module.test_activations
        device = pl_module.device

        for dataloader_idx, acts in all_acts.items():
            Za = _resolve_modality_tensors(acts, "audio", device, trainer)
            Zt = _resolve_modality_tensors(acts, "text", device, trainer)

            if Za is None or Zt is None:
                log.warning(f"[ModalityClassifierCallback] Missing modality activations for mode={mode}, dataloader_idx={dataloader_idx}.")
                continue

            result = self._compute(Za, Zt, max_samples=self.max_samples, test_fraction=self.test_fraction)
            if result is None:
                continue

            dataset_name = _get_dataset_name(trainer, dataloader_idx, mode)
            log_prefix = f"Modality Classifier/{dataset_name}"
            pl_module.log(f"{log_prefix}/mean_acc", result["mean_acc"], prog_bar=True, sync_dist=True)
            pl_module.log(f"{log_prefix}/energy_weighted_mean_acc", result["energy_weighted_mean_acc"], sync_dist=True)

            self._log_accuracy_histograms(trainer, result["accuracies"], result["E_np"], mode, dataset_name)
            self._log_per_neuron_accuracy_barplot(trainer, result["accuracies"], result["E_np"], mode, dataset_name)

    def _log_accuracy_histograms(
        self,
        trainer: Trainer,
        accuracies,
        E,
        mode: str,
        dataset_name: str,
    ):
        """
        Log two histograms of per-concept classification accuracies:
        - Unweighted: proportion of concepts in each accuracy bin.
        - Energy-weighted: proportion of energy in each accuracy bin.
        """
        try:
            import numpy as np
            import plotly.graph_objects as go
        except ImportError:
            log.debug("numpy/plotly not available, skipping accuracy histograms")
            return

        bins = np.linspace(0.0, 1.0, 51)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Unweighted
        counts, _ = np.histogram(accuracies, bins=bins)
        total = max(counts.sum(), 1e-12)
        density_unweighted = (counts / total).tolist()

        # Energy-weighted
        counts_w, _ = np.histogram(accuracies, bins=bins, weights=E)
        total_E = max(E.sum(), 1e-12)
        density_weighted = (counts_w / total_E).tolist()

        layout_base = dict(
            xaxis_title="Per-Concept Classification Accuracy",
            yaxis_title="Proportion",
            template="plotly_white",
            bargap=0.1,
        )

        fig_unweighted = go.Figure()
        fig_unweighted.add_trace(
            go.Bar(
                x=bin_centers.tolist(),
                y=density_unweighted,
                marker=dict(color="teal", line=BAR_MARKER_LINE),
                width=BAR_WIDTH,
            )
        )
        fig_unweighted.update_layout(
            title=f"Per-Concept Modality Classification Accuracy — Unweighted ({mode})",
            **layout_base,
        )
        fig_unweighted.update_yaxes(rangemode="tozero")
        _log_plotly_to_wandb(
            trainer,
            f"Modality Classifier/{dataset_name}/acc_histogram_unweighted",
            fig_unweighted,
        )

        fig_weighted = go.Figure()
        fig_weighted.add_trace(
            go.Bar(
                x=bin_centers.tolist(),
                y=density_weighted,
                marker=dict(color="orange", line=BAR_MARKER_LINE),
                width=BAR_WIDTH,
            )
        )
        fig_weighted.update_layout(
            title=f"Per-Concept Modality Classification Accuracy — Energy-Weighted ({mode})",
            **layout_base,
        )
        fig_weighted.update_yaxes(rangemode="tozero")
        _log_plotly_to_wandb(
            trainer,
            f"Modality Classifier/{dataset_name}/acc_histogram_energy_weighted",
            fig_weighted,
        )

    def _log_per_neuron_accuracy_barplot(
        self,
        trainer: Trainer,
        accuracies,
        E,
        mode: str,
        dataset_name: str,
    ):
        """
        Log a barplot showing accuracy for each neuron/concept.
        X-axis: neuron index, Y-axis: classification accuracy.
        """
        try:
            import numpy as np
            import plotly.graph_objects as go
        except ImportError:
            log.debug("numpy/plotly not available, skipping per-neuron accuracy barplot")
            return

        n_concepts = len(accuracies)
        x_norm = _normalized_neuron_x(n_concepts)

        # Sort by accuracy (descending) for better visualization
        sorted_idx = np.argsort(accuracies)[::-1]
        sorted_accuracies = accuracies[sorted_idx]
        sorted_energies = E[sorted_idx]

        # Color by energy (normalized)
        E_norm = sorted_energies / max(sorted_energies.max(), 1e-12)
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x_norm,
                y=sorted_accuracies.tolist(),
                width=0.001,
                marker=dict(
                    color=E_norm.tolist(),
                    colorscale="Viridis",
                    colorbar=dict(title="Normalized<br>Energy"),
                    line=BAR_MARKER_LINE,
                ),
                hovertemplate="Neuron: %{x:.4f}<br>Accuracy: %{y:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Per-Neuron Classification Accuracy — sorted by accuracy ({mode})",
            xaxis_title="Neuron Index (normalized 0–1, sorted by accuracy)",
            yaxis_title="Classification Accuracy",
            template="plotly_white",
            yaxis=dict(range=[0.5, 1]),
            height=500,
        )
        _log_plotly_to_wandb(
            trainer,
            f"Modality Classifier/{dataset_name}/per_neuron_accuracy",
            fig,
        )


# -----------------------------------------------------------------------------
# CrossRunStabilityCallback (TEST ONLY)
# -----------------------------------------------------------------------------
class CrossRunStabilityCallback(BaseCallback):
    """
    TEST ONLY.

    Stability between current dictionary D0 and checkpoint dictionary Dk using Hungarian matching
    over cosine similarities (max average matched similarity).

    Optionally also computes stability restricted to top-K concepts by test energy.

    Args:
        checkpoint_paths: list of s3://... or local paths
        topk_by_energy: if not None, compute additional stability on top-K energetic concepts
        n_curve_points: number of points to sample for stability vs K plots
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        topk_by_energy: Optional[int] = 512,
        prefix: str = "stability",
        n_curve_points: int = 50,
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.checkpoint_paths = checkpoint_paths
        self.topk_by_energy = topk_by_energy
        self.prefix = prefix
        self.n_curve_points = n_curve_points

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        return  # explicitly disabled

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self._compute(trainer, pl_module)

    @torch.no_grad()
    def _compute(self, trainer: Trainer, pl_module: LightningModule):
        if not self.checkpoint_paths:
            log.warning("[CrossRunStabilityCallback] No checkpoint paths provided.")
            return

        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise ImportError(
                "CrossRunStabilityCallback requires scipy (scipy.optimize.linear_sum_assignment). "
                f"Install scipy or replace Hungarian implementation. Original error: {e}"
            )

        device = pl_module.device

        # Current dictionary
        D0 = get_dictionary_from_lightningsae(pl_module).to(device)
        D0 = F.normalize(D0, dim=-1)  # [C, d]
        n_concepts = D0.shape[0]

        # Get energy for ranking concepts
        E = None
        if hasattr(pl_module, "_last_energy") and pl_module._last_energy.get("combined") is not None:
            E = pl_module._last_energy["combined"].to(device)

        # Otherwise compute from test activations
        if E is None:
            all_test_acts = getattr(pl_module, "test_activations", None)
            if all_test_acts:
                for dataloader_idx, acts in all_test_acts.items():
                    if acts.get("audio") and acts.get("text"):
                        if isinstance(acts["audio"], list):
                            Za = torch.cat([a.to(device) for a in acts["audio"]], dim=0)
                        else:
                            Za = acts["audio"].to(device)
                        
                        if isinstance(acts["text"], list):
                            Zt = torch.cat([a.to(device) for a in acts["text"]], dim=0)
                        else:
                            Zt = acts["text"].to(device)
                        
                        Za = gather_tensor_if_distributed(Za, trainer)
                        Zt = gather_tensor_if_distributed(Zt, trainer)
                        E = 0.5 * (Za.mean(0) + Zt.mean(0))
                        break

        # Get sorted indices by energy (descending)
        if E is not None:
            sorted_indices = torch.argsort(E, descending=True)
            E_sorted = E[sorted_indices]
            total_energy = E_sorted.sum().clamp_min(1e-12)
            cumulative_energy = torch.cumsum(E_sorted, dim=0) / total_energy
        else:
            sorted_indices = torch.arange(n_concepts, device=device)
            cumulative_energy = None

        # Define K values to sample for the stability curves
        k_values = self._get_k_values(n_concepts)

        def stability_hungarian(A: torch.Tensor, B: torch.Tensor) -> float:
            sim = (A @ B.t()).detach().cpu().numpy()
            cost = -sim
            r, c = linear_sum_assignment(cost)
            return float(sim[r, c].mean())

        stabs_all = []
        stabs_top = []
        # For curve plots: stability_per_k[checkpoint_idx] = list of stabilities for each k
        stability_per_k_all = []

        for p in self.checkpoint_paths:
            m = copy.deepcopy(pl_module).to("cpu")
            sd = load_state_dict_any(p, map_location="cpu")
            m.load_state_dict(sd, strict=False)

            Dk = get_dictionary_from_lightningsae(m).to(device)
            Dk = F.normalize(Dk, dim=-1)

            # Full stability
            stab = stability_hungarian(D0, Dk)
            stabs_all.append(stab)
            pl_module.log(f"test/{self.prefix}_vs_{_safe_name(p)}", stab, sync_dist=True)

            # Top-K stability (single value)
            if self.topk_by_energy is not None and E is not None:
                k = min(self.topk_by_energy, n_concepts)
                top_idx = sorted_indices[:k]
                stab_top = stability_hungarian(D0[top_idx], Dk[top_idx])
                stabs_top.append(stab_top)
                pl_module.log(
                    f"test/{self.prefix}_top{k}_vs_{_safe_name(p)}",
                    stab_top,
                    sync_dist=True,
                )

            # Compute stability for each K value (for curve plots)
            stabs_for_k = []
            for k in k_values:
                top_idx_k = sorted_indices[:k]
                stab_k = stability_hungarian(D0[top_idx_k], Dk[top_idx_k])
                stabs_for_k.append(stab_k)
            stability_per_k_all.append(stabs_for_k)

        # Log mean stability
        pl_module.log(
            f"test/{self.prefix}_mean",
            float(sum(stabs_all) / max(len(stabs_all), 1)),
            prog_bar=True,
            sync_dist=True,
        )
        if stabs_top:
            k = min(self.topk_by_energy, n_concepts)
            pl_module.log(
                f"test/{self.prefix}_top{k}_mean",
                float(sum(stabs_top) / max(len(stabs_top), 1)),
                prog_bar=True,
                sync_dist=True,
            )

        # Compute mean stability across checkpoints for each K
        if stability_per_k_all:
            import numpy as np
            stab_arr = np.array(stability_per_k_all)  # [n_checkpoints, n_k_values]
            mean_stab_per_k = stab_arr.mean(axis=0).tolist()

            # Plot 1: Stability vs Number of Concepts
            self._log_stability_vs_num_concepts_plot(
                trainer, k_values, mean_stab_per_k
            )

            # Plot 2: Stability vs Cumulative Energy (energy-weighted)
            if cumulative_energy is not None:
                cum_energy_at_k = [cumulative_energy[k - 1].item() for k in k_values]
                self._log_stability_vs_cumulative_energy_plot(
                    trainer, cum_energy_at_k, mean_stab_per_k
                )

    def _get_k_values(self, n_concepts: int) -> List[int]:
        """Generate K values to sample for stability curves (log-spaced for better resolution at low K)."""
        import numpy as np
        # Use log spacing with more points at lower K values
        n_points = min(self.n_curve_points, n_concepts)
        k_vals = np.unique(np.geomspace(1, n_concepts, n_points).astype(int))
        # Ensure we have at least 1 and n_concepts
        k_vals = sorted(set([1] + k_vals.tolist() + [n_concepts]))
        return k_vals

    def _log_stability_vs_num_concepts_plot(
        self,
        trainer: Trainer,
        k_values: List[int],
        mean_stabilities: List[float],
    ):
        """Plot stability as a function of number of concepts (ranked by energy)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.debug("plotly not available, skipping stability vs num concepts plot")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values,
            y=mean_stabilities,
            mode="lines+markers",
            name="Mean Stability",
            line=dict(color="green", width=2),
            marker=dict(size=4),
        ))
        fig.update_layout(
            title="Stability vs Number of Concepts (ranked by energy)",
            xaxis_title="Number of Top Concepts (by energy)",
            yaxis_title="Stability (Hungarian matching)",
            yaxis_range=[0, 1.05],
            xaxis_type="log",
            template="plotly_white",
        )

        plot_key = f"test/{self.prefix}_vs_num_concepts_plot"
        _log_plotly_to_wandb(trainer, plot_key, fig)

    def _log_stability_vs_cumulative_energy_plot(
        self,
        trainer: Trainer,
        cumulative_energy: List[float],
        mean_stabilities: List[float],
    ):
        """Plot stability as a function of cumulative energy fraction (energy-weighted view)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.debug("plotly not available, skipping stability vs cumulative energy plot")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_energy,
            y=mean_stabilities,
            mode="lines+markers",
            name="Mean Stability",
            line=dict(color="purple", width=2),
            marker=dict(size=4),
        ))
        fig.update_layout(
            title="Stability vs Cumulative Energy Fraction",
            xaxis_title="Cumulative Energy Fraction",
            yaxis_title="Stability (Hungarian matching)",
            xaxis_range=[0, 1.05],
            yaxis_range=[0, 1.05],
            template="plotly_white",
        )

        plot_key = f"test/{self.prefix}_vs_cumulative_energy_plot"
        _log_plotly_to_wandb(trainer, plot_key, fig)


# -----------------------------------------------------------------------------
# SimLoggerCallback
# -----------------------------------------------------------------------------
class SimLoggerCallback(BaseCallback):
    """
    Logs distributions of pairwise cosine similarities between embeddings:
      - audio-audio (blue)
      - text-text (red)
      - text-audio off-diagonal (purple)
      - text-audio diagonal / matched pairs (dark purple)

    Reads embeddings from pl_module.val_embeddings / pl_module.test_embeddings
    (populated by LightningSAE._eval_step).
    """

    def __init__(
        self,
        enable_on_validation: bool = True,
        enable_on_test: bool = True,
        every_n_steps: int = None,
        every_n_epochs: int = 1,
        max_samples: int = 2048,
        n_bins: int = 50,
        prefix: str = "similarity",
    ):
        super().__init__(every_n_steps=every_n_steps, every_n_epochs=every_n_epochs)
        self.enable_on_validation = enable_on_validation
        self.enable_on_test = enable_on_test
        self.max_samples = max_samples
        self.n_bins = n_bins
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
        audio_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Pure computation — no trainer / pl_module / logging.

        Args:
            audio_emb: [N_a, D] or None
            text_emb:  [N_t, D] or None

        Returns:
            dict mapping label -> 1D tensor of cosine similarities.
            Labels: 'audio-audio', 'text-text', 'text-audio (matched)',
                    'text-audio (off-diag)'.
        """
        def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T

        sims = {}

        if audio_emb is not None and audio_emb.size(0) > 1:
            sim_aa = cosine_sim_matrix(audio_emb, audio_emb)
            triu_idx = torch.triu_indices(sim_aa.size(0), sim_aa.size(1), offset=1)
            sims["audio-audio"] = sim_aa[triu_idx[0], triu_idx[1]]

        if text_emb is not None and text_emb.size(0) > 1:
            sim_tt = cosine_sim_matrix(text_emb, text_emb)
            triu_idx = torch.triu_indices(sim_tt.size(0), sim_tt.size(1), offset=1)
            sims["text-text"] = sim_tt[triu_idx[0], triu_idx[1]]

        if audio_emb is not None and text_emb is not None:
            sim_ta = cosine_sim_matrix(text_emb, audio_emb)
            n_diag = min(sim_ta.shape[0], sim_ta.shape[1])
            sims["text-audio (matched)"] = torch.diag(sim_ta[:n_diag, :n_diag])
            mask = torch.ones_like(sim_ta, dtype=torch.bool)
            for i in range(n_diag):
                mask[i, i] = False
            sims["text-audio (off-diag)"] = sim_ta[mask]

        return sims

    @torch.no_grad()
    def _compute_and_log(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        """Gather data from module, call _compute, then log."""
        all_embs = getattr(pl_module, f"{mode}_embeddings", {})
        device = pl_module.device

        for dataloader_idx, embs in all_embs.items():
            audio_emb = _resolve_modality_tensors(embs, "audio", device, trainer)
            text_emb = _resolve_modality_tensors(embs, "text", device, trainer)

            if audio_emb is None and text_emb is None:
                log.warning(f"[SimLoggerCallback] No embeddings for mode={mode}, dataloader_idx={dataloader_idx}.")
                continue

            # Move to CPU and subsample
            if audio_emb is not None:
                audio_emb = audio_emb.cpu()
                if audio_emb.size(0) > self.max_samples:
                    audio_emb = audio_emb[torch.randperm(audio_emb.size(0))[: self.max_samples]]
            if text_emb is not None:
                text_emb = text_emb.cpu()
                if text_emb.size(0) > self.max_samples:
                    text_emb = text_emb[torch.randperm(text_emb.size(0))[: self.max_samples]]

            sims = self._compute(audio_emb, text_emb)
            if not sims:
                continue

            dataset_name = _get_dataset_name(trainer, dataloader_idx, mode)
            self._log_similarity_figure(trainer, sims, mode, dataset_name)

    def _log_similarity_figure(
        self,
        trainer: Trainer,
        sims: Dict[str, torch.Tensor],
        mode: str,
        dataset_name: str,
    ):
        """Build and log plotly figure from pre-computed similarity tensors."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            log.debug("plotly not available, skipping similarity distribution plot")
            return

        colors = {
            "audio-audio": "rgba(0, 0, 255, 0.6)",
            "text-text": "rgba(255, 0, 0, 0.6)",
            "text-audio (off-diag)": "rgba(128, 0, 128, 0.5)",
            "text-audio (matched)": "rgba(75, 0, 130, 0.8)",
        }

        fig = go.Figure()
        for label, vals in sims.items():
            fig.add_trace(go.Histogram(
                x=vals.cpu().numpy().tolist(),
                name=label,
                opacity=0.7,
                marker_color=colors.get(label, "gray"),
                nbinsx=self.n_bins,
                histnorm="probability density",
            ))

        fig.update_layout(
            title=f"Embedding Cosine Similarity Distributions ({mode})",
            xaxis_title="Cosine Similarity",
            yaxis_title="Density",
            barmode="overlay",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        _log_plotly_to_wandb(trainer, f"Similarity/{dataset_name}/distributions", fig)
