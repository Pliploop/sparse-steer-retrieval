"""Loss functions for SAE training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from steerable_retrieval.models.sae.penalties import StepFunction


def r2_score(reconstruction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Coefficient of determination (variance-weighted, aggregated over all outputs).

    ``1 - SS_res / SS_tot`` with ``target`` as ground truth. Logged as a reconstruction
    quality metric during SAE training.
    """
    reconstruction = reconstruction.float()
    target = target.float()
    ss_res = (target - reconstruction).pow(2).sum()
    ss_tot = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
    return 1.0 - ss_res / (ss_tot + eps)


class SAELoss(nn.Module):
    """
    Base loss function for SAE training.
    Computes loss from reconstruction, activations, and target.
    """

    def __init__(self, l1_coeff=1e-3, l2_coeff=1.0):
        super().__init__()
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff

    def forward(self, reconstruction, activations, target, matryoshka=None, num_batches_not_active=None, decoder=None, **kwargs):
        """
        Compute SAE loss.

        Args:
            reconstruction: Reconstructed features [batch, ..., act_size]
            activations: Feature activations [batch, ..., dict_size]
            target: Target features [batch, ..., act_size]
            matryoshka: Matryoshka module (optional, for compatibility)
            num_batches_not_active: Tracking of inactive features (optional, for compatibility)
            decoder: SAE decoder (optional, used by TopK/BatchTopK aux loss)

        Returns:
            Dictionary with loss components
        """
        # L2 reconstruction loss (MSE scaled by l2_coeff)
        rec_err = (reconstruction.float() - target.float()).pow(2)
        mse = rec_err.mean()
        l2_loss = self.l2_coeff * mse

        r2 = r2_score(reconstruction, target)
        # dead_codes_frac = dead_codes(activations)
        # ndn = dead_codes_frac.mean() if hasattr(dead_codes_frac, "mean") else dead_codes_frac

        alive = (activations > 0).any(dim=0)
        dead_ratio = 1 - alive.float().mean()

        # L1 sparsity loss
        l1_norm = activations.float().abs().sum(-1).mean()
        l1_loss = self.l1_coeff * l1_norm

        # L0 norm (number of active features)
        l0_norm = (activations > 0).float().sum(-1).mean()

        # Total loss
        loss = l2_loss + l1_loss

        return {
            'loss': loss,
            'l2_loss': l2_loss,
            'l1_loss': l1_loss,
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'mse': mse.detach(),
            'r2': r2,
            'ndn': dead_ratio.detach(),
        }


class VanillaSAELoss(SAELoss):
    """Loss for VanillaSAE - standard L1 + L2."""
    pass


class TopKSAELoss(SAELoss):
    """Loss for TopKSAE - includes auxiliary loss for dead features."""

    def __init__(self, l1_coeff=1e-3, l2_coeff=1.0, aux_penalty=1.0, top_k_aux=10, n_batches_to_dead=10, orthogonal_loss=0.0):
        super().__init__(l1_coeff, l2_coeff)
        self.aux_penalty = aux_penalty
        self.top_k_aux = top_k_aux
        self.n_batches_to_dead = n_batches_to_dead
        self.orthogonal_coeff = orthogonal_loss
    def forward(self, reconstruction, activations, target,
            num_batches_not_active=None, matryoshka=None, decoder=None,
            pre_penalty_activations=None):
        loss_dict = super().forward(reconstruction, activations, target)
        aux_loss = self._get_auxiliary_loss(
            target=target,
            reconstruction=reconstruction,
            activations=activations,
            # num_batches_not_active=num_batches_not_active,
            decoder=decoder,
            pre_penalty_activations=pre_penalty_activations,
        )
        loss_dict['aux_loss'] = aux_loss
        loss_dict['orthogonal_loss'] = self.offdiag_gram_loss(D = decoder.W_dec)
        loss_dict['loss'] = loss_dict['loss'] + aux_loss + loss_dict['orthogonal_loss'] * self.orthogonal_coeff
        return loss_dict
    

    def _get_auxiliary_loss(
        self,
        target,
        reconstruction,
        activations,
        decoder=None,
        pre_penalty_activations=None,
    ):
        # Match their behavior: if missing inputs, aux=0
        if decoder is None or pre_penalty_activations is None or self.aux_penalty <= 0:
            return torch.tensor(0.0, device=target.device, dtype=target.dtype)

        # IMPORTANT: order matters (their comment) => residual = x - x_hat
        residual = (target - reconstruction)

        # Their AuxK uses relu(pre) and removes chosen codes by subtracting post-TopK codes
        aux_src = F.relu(pre_penalty_activations) - activations

        # Choose top half of *non-chosen* activations (after subtraction)
        k = max(1, aux_src.shape[-1] // 2)
        topk = torch.topk(aux_src, k=k, dim=-1)

        aux_codes = torch.zeros_like(aux_src).scatter(-1, topk.indices, topk.values)

        # Predict residual using full dictionary; no bias term
        residual_hat = aux_codes @ decoder.W_dec  # [B, act_size]

        aux_mse = (residual - residual_hat).pow(2).mean()
        return self.aux_penalty * aux_mse


    @staticmethod
    def offdiag_gram_loss(
        D: torch.Tensor,
        atoms_dim: int | None = None,
        normalize: bool = True,
        squared: bool = True,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Penalize off-diagonal entries of the (optionally normalized) Gram matrix.

        This encourages *incoherence* (low pairwise cosine similarity) between atoms,
        without requiring strict orthogonality (which is infeasible when overcomplete).

        Parameters
        ----------
        D : Tensor
            Dictionary/decoder matrix, shape [A, d] or [d, A] where A=#atoms/features.
            Examples:
            - If W_dec is [dict_size, act_size], then atoms_dim=0.
            - If W_dec is [act_size, dict_size], then atoms_dim=1.
        atoms_dim : {0,1} or None
            Which dimension indexes atoms. If None, we infer it by assuming atoms are
            the larger dimension (common in overcomplete SAEs).
        normalize : bool
            If True, L2-normalize each atom before computing Gram (cosine coherence).
        squared : bool
            If True use L2 penalty on off-diagonals (sum of squares). If False use L1.
        reduction : {"mean","sum"}
            How to reduce over off-diagonal entries.
        eps : float
            Numerical stability for normalization.

        Returns
        -------
        loss : Tensor
            Scalar tensor.
        """
        A = D.clone()

        if normalize:
            A = A / (A.norm(dim=1, keepdim=True).clamp_min(eps))

        # Gram: [A, A]
        G = A @ A.t()

        # Off-diagonal mask
        n = G.shape[0]
        if n <= 1:
            return G.new_zeros(())
        off = ~torch.eye(n, dtype=torch.bool, device=G.device)

        vals = G[off]
        if squared:
            vals = vals * vals
        else:
            vals = vals.abs()

        if reduction == "mean":
            return vals.mean()
        elif reduction == "sum":
            return vals.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")


class BatchTopKSAELoss(TopKSAELoss):
    """Loss for BatchTopKSAE - same as TopK but with batch-level sparsity."""
    pass


class JumpReLUSAELoss(SAELoss):
    """Loss for JumpReLUSAE - uses L0 norm instead of L1."""
    
    def __init__(self, l1_coeff=1e-3, l2_coeff=1.0, bandwidth=0.1):
        super().__init__(l1_coeff, l2_coeff)
        self.bandwidth = bandwidth
    
    def forward(self, reconstruction, activations, target, log_threshold=None, matryoshka=None, num_batches_not_active=None, decoder=None):
        """
        Compute JumpReLU SAE loss.
        
        Args:
            reconstruction: Reconstructed features
            activations: Feature activations
            target: Target features
            log_threshold: Log threshold parameter from JumpReLU (optional)
            matryoshka: Matryoshka module (optional, for compatibility)
            num_batches_not_active: Tracking of inactive features (optional, for compatibility)
        """
        # L2 reconstruction loss
        rec_err = (reconstruction.float() - target.float()).pow(2)
        mse = rec_err.mean()
        l2_loss = self.l2_coeff * mse
        target_centered = target.float() - target.float().mean(dim=0)
        baseline_mse = (target_centered.pow(2)).mean().clamp_min(1e-12)
        fvu = (mse / baseline_mse).detach()
        ndn = (activations.abs().sum(dim=0) == 0).float().mean().detach()

        # L0 norm using StepFunction if log_threshold is provided
        if log_threshold is not None:
            l0 = StepFunction.apply(activations, log_threshold, self.bandwidth).sum(dim=-1).mean()
        else:
            l0 = (activations > 0).float().sum(-1).mean()

        l0_loss = self.l1_coeff * l0
        l1_loss = l0_loss  # For JumpReLU, L1 loss is the same as L0 loss

        loss = l2_loss + l1_loss

        return {
            'loss': loss,
            'l2_loss': l2_loss,
            'l1_loss': l1_loss,
            'l0_norm': l0,
            'l1_norm': l0,
            'mse': mse.detach(),
            'fvu': fvu,
            'ndn': ndn,
        }

class MatryoshkaBatchTopKSAELoss(SAELoss):
    """Loss for Matryoshka BatchTopK SAE with progressive reconstruction losses.

    Computes intermediate reconstructions by progressively decoding each
    feature group through ``decoder.W_dec`` slices.  Supports two weighting
    strategies from Zaigrajew et al. (2025):

    * **uniform** (UW): ``alpha_i = 1`` for all granularity levels.
    * **reverse** (RW): ``alpha_i = h - i + 1``, giving higher weight to
      sparser (earlier) levels so the model prioritises reconstruction
      quality at low feature counts.

    Also includes a TopK-style auxiliary loss for dead features.
    """

    def __init__(self, group_sizes, l1_coeff=1e-3, l2_coeff=1.0,
                 aux_penalty=1.0, top_k_aux=10, n_batches_to_dead=10,
                 weighting='uniform'):
        super().__init__(l1_coeff, l2_coeff)
        self.group_sizes = group_sizes
        self.group_indices = [0] + torch.cumsum(torch.tensor(group_sizes), dim=0).tolist()
        self.aux_penalty = aux_penalty
        self.top_k_aux = top_k_aux
        self.n_batches_to_dead = n_batches_to_dead

        h = len(group_sizes)
        if weighting == 'uniform':
            self.alphas = [1.0] * h
        elif weighting == 'reverse':
            self.alphas = [float(h - i) for i in range(h)]
        else:
            raise ValueError(f"Unknown weighting strategy '{weighting}'. Use 'uniform' or 'reverse'.")
        self.alpha_sum = sum(self.alphas)

    def forward(self, reconstruction, activations, target,
                decoder=None, pre_penalty_activations=None,
                num_batches_not_active=None, **kwargs):
        """
        Compute Matryoshka BatchTopK loss with progressive intermediate losses.

        Args:
            reconstruction: Final reconstructed features [batch, act_size]
            activations: Sparse feature activations [batch, dict_size]
            target: Target features [batch, act_size]
            decoder: SAE decoder (required -- provides W_dec and b_dec)
            pre_penalty_activations: Pre-activation features before ReLU/TopK [batch, dict_size]
            num_batches_not_active: Per-feature inactivity counter (optional)

        Returns:
            Dictionary with loss components.
        """
        target_f = target.float()

        # -- Intermediate reconstruction losses (progressive decoding) -------
        b_dec = decoder.b_dec if decoder is not None else torch.zeros_like(target_f[0])

        # Baseline: reconstruction using only b_dec (no features active)
        baseline_l2 = self.l2_coeff * (b_dec - target_f).pow(2).mean()

        x_recon = b_dec  # accumulates progressive reconstruction
        intermediate_l2s = []
        for i in range(len(self.group_sizes)):
            start = self.group_indices[i]
            end = self.group_indices[i + 1]
            x_recon = activations[..., start:end] @ decoder.W_dec[start:end] + x_recon
            intermediate_l2s.append(
                self.l2_coeff * (x_recon.float() - target_f).pow(2).mean()
            )

        # Weighted mean L2: baseline + weighted intermediates, divided by
        # (1 + sum_of_weights) to match the reference's (h+1) averaging.
        weighted_l2 = baseline_l2 + sum(
            a * l for a, l in zip(self.alphas, intermediate_l2s)
        )
        mean_l2 = weighted_l2 / (1.0 + self.alpha_sum)

        l2_stack = torch.stack(intermediate_l2s)
        min_l2 = l2_stack.min()
        max_l2 = l2_stack.max()

        # -- Final-reconstruction metrics (mse, r2, ndn) --------------------
        mse = (reconstruction.float() - target_f).pow(2).mean()
        r2 = r2_score(reconstruction, target)
        alive = (activations > 0).any(dim=0)
        ndn = 1 - alive.float().mean()

        # -- Sparsity losses -------------------------------------------------
        l1_norm = activations.float().abs().sum(-1).mean()
        l1_loss = self.l1_coeff * l1_norm
        l0_norm = (activations > 0).float().sum(-1).mean()

        # -- Auxiliary loss for dead features ---------------------------------
        aux_loss = self._get_auxiliary_loss(
            target=target,
            reconstruction=reconstruction,
            activations=activations,
            decoder=decoder,
            pre_penalty_activations=pre_penalty_activations,
        )

        loss = mean_l2 + l1_loss + aux_loss

        return {
            'loss': loss,
            'l2_loss': mean_l2,
            'min_l2_loss': min_l2.detach(),
            'max_l2_loss': max_l2.detach(),
            'l1_loss': l1_loss,
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'aux_loss': aux_loss,
            'mse': mse.detach(),
            'r2': r2,
            'ndn': ndn.detach(),
        }

    def _get_auxiliary_loss(self, target, reconstruction, activations,
                            decoder=None, pre_penalty_activations=None):
        """TopK-style auxiliary loss: predict the residual using dead features."""
        if decoder is None or pre_penalty_activations is None or self.aux_penalty <= 0:
            return torch.tensor(0.0, device=target.device, dtype=target.dtype)

        residual = target - reconstruction
        aux_src = F.relu(pre_penalty_activations) - activations

        k = max(1, aux_src.shape[-1] // 2)
        topk = torch.topk(aux_src, k=k, dim=-1)
        aux_codes = torch.zeros_like(aux_src).scatter(-1, topk.indices, topk.values)

        residual_hat = aux_codes @ decoder.W_dec
        aux_mse = (residual - residual_hat).pow(2).mean()
        return self.aux_penalty * aux_mse
