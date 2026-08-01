"""SAE encoder classes that return only activations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from overcomplete.optimization import batched_matrix_nnls
except ImportError:
    batched_matrix_nnls = None

from steerable_retrieval.models.sae.penalties import (
    VanillaPenalty, TopKPenalty, BatchTopKPenalty,
    MatryoshkaBatchTopKPenalty, JumpReLU,
)


class SAEEncoder(nn.Module):
    """Base SAE encoder that takes audio/text and returns activations."""

    def __init__(self, act_size, dict_size, device='cuda', 
                 input_unit_norm=False, seed=None, concat_dim=-1, **kwargs):
        super().__init__()
        self.act_size = act_size
        self.dict_size = dict_size
        self.device = device
        self.dtype = torch.float32  # Automatic dtype
        self.input_unit_norm = input_unit_norm
        self.concat_dim = concat_dim
        self.cfg = {
            "act_size": act_size,
            "dict_size": dict_size,
            "device": device,
            "dtype": self.dtype,
            "input_unit_norm": input_unit_norm,
            **kwargs
        }
        
        if seed is not None:
            torch.manual_seed(seed)

        # Encoder weights and biases
        self.b_dec = nn.Parameter(torch.zeros(act_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(act_size, dict_size)
            )
        )
        
        # Track inactive features
        self.num_batches_not_active = torch.zeros((dict_size,)).to(device)

        self.to(self.dtype).to(device)
        
        # Required by SAE class
        self.out_channels = dict_size

    def forward(self, x):
        """
        Forward pass through SAE encoder.
        
        Args:
            x: Input features tensor [batch, ..., act_size]
               Can be audio or text features (already encoded by audio_encoder/text_encoder)
            
        Returns:
            activations: Feature activations [batch, ..., dict_size]
        """
        # Encode to get activations
        pre, z = self._encode(x)

        self.update_inactive_features(z)
        
        return pre, z

    def _encode(self, x):
        """Internal encode method - to be overridden by subclasses."""
        raise NotImplementedError

    def preprocess_input(self, x):
        if self.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            # unit norm the input
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
            return x, x_mean, None
        else:
            return x, None, None

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class VanillaSAEEncoder(SAEEncoder):
    """Vanilla SAE encoder with standard ReLU activation."""

    def _encode(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        pre_activations = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_activations)
        return pre_activations, acts


class TopKSAEEncoder(SAEEncoder):
    """TopK SAE encoder - keeps top k activations per sample."""

    def __init__(self, act_size, dict_size, top_k, device='cuda', 
                 input_unit_norm=False, seed=None, concat_dim=-1, **kwargs):
        super().__init__(act_size, dict_size, device, input_unit_norm, seed, concat_dim, top_k=top_k, **kwargs)
        self.top_k = top_k
        self.penalty = TopKPenalty(top_k)

    def _init_decoder_weights(self, W_dec):
        # initialize the encoder weight to the transpose of the decoder weight
        self.W_enc.data.copy_(W_dec.T)

    def _encode(self, x):
        x, _, _ = self.preprocess_input(x)
        x_cent = x - self.b_dec
        pre_activations = x_cent @ self.W_enc
        acts = F.relu(pre_activations)
        acts = self.penalty(acts)
        return pre_activations, acts


class BatchTopKSAEEncoder(SAEEncoder):
    """
    Batch Top-k SAE encoder (Bussmann et al., 2024).

    Retains only the top-k global activations across the entire batch.
    During training, the k-th highest activation (over the flattened batch)
    is used as the threshold and a running average is maintained with
    exponential momentum.  At eval time the running threshold is used,
    making inference deterministic and batch-size independent.
    """

    def __init__(self, act_size, dict_size, top_k, threshold_momentum=0.9,
                 device='cuda', input_unit_norm=False, seed=None, concat_dim=-1, **kwargs):
        super().__init__(act_size, dict_size, device, input_unit_norm, seed, concat_dim,
                         top_k=top_k, threshold_momentum=threshold_momentum, **kwargs)
        self.top_k = top_k
        self.penalty = BatchTopKPenalty(top_k, threshold_momentum=threshold_momentum)
        self.register_buffer("running_threshold", None)
        self.threshold_momentum = threshold_momentum
    def _init_decoder_weights(self, W_dec):
        self.W_enc.data.copy_(W_dec.T)

    def _encode(self, x):
        x, _, _ = self.preprocess_input(x)
        x_cent = x - self.b_dec
        pre_activations = x_cent @ self.W_enc
        acts = F.relu(pre_activations)
        threshold = self.penalty._get_threshold(acts)
        threshold = self._update_threshold(threshold)
        acts = self.penalty(acts, threshold)
        return pre_activations, acts

    def _rectify(self, acts):
        threshold = self.penalty._get_threshold(acts)
        threshold = self._update_threshold(threshold)
        acts = self.penalty(acts, threshold)
        return acts
    
    def _update_threshold(self, threshold):
        if self.training:
            if self.running_threshold is None:
                self.running_threshold = threshold.detach()
            else:
                self.running_threshold = (
                    self.threshold_momentum * self.running_threshold
                    + (1 - self.threshold_momentum) * threshold.detach()
                )
            return threshold
        else:
            if self.running_threshold is None:
                self.running_threshold = threshold.detach()
                return threshold
            else:
                return self.running_threshold
    

    


class MatryoshkaBatchTopKSAEEncoder(BatchTopKSAEEncoder):
    """
    Matryoshka Batch Top-k SAE encoder.

    Combines nested feature groups (Matryoshka) with global batch-level TopK
    sparsity.  ``group_sizes`` defines the nested structure and implicitly sets
    ``dict_size = sum(group_sizes)``.  Features beyond the currently active
    groups are zeroed out by the penalty.
    """

    def __init__(self, act_size, group_sizes, top_k, threshold_momentum=0.9,
                 device='cuda', input_unit_norm=False, seed=None, concat_dim=-1, **kwargs):
        dict_size = sum(group_sizes)
        super().__init__(act_size, dict_size, top_k, threshold_momentum,
                         device, input_unit_norm, seed, concat_dim, **kwargs)
        self.group_sizes = group_sizes
        # Replace the BatchTopKPenalty created by parent with Matryoshka variant
        self.penalty = MatryoshkaBatchTopKPenalty(top_k, group_sizes, threshold_momentum)


class JumpReLUSAEEncoder(SAEEncoder):
    """JumpReLU SAE encoder with learnable thresholds."""

    def __init__(self, act_size, dict_size, bandwidth=0.1, device='cuda', 
                 input_unit_norm=False, use_pre_enc_bias=False, seed=None, concat_dim=-1, **kwargs):
        super().__init__(act_size, dict_size, device, input_unit_norm, seed, concat_dim, 
                        bandwidth=bandwidth, use_pre_enc_bias=use_pre_enc_bias, **kwargs)
        self.bandwidth = bandwidth
        self.use_pre_enc_bias = use_pre_enc_bias
        self.jumprelu = JumpReLU(feature_size=dict_size, bandwidth=bandwidth, device=device)

    def _encode(self, x):
        x, _, _ = self.preprocess_input(x)
        
        if self.use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = x @ self.W_enc + self.b_enc
        acts = self.jumprelu(acts)
        return pre_activations, acts


class OMPEncoder(SAEEncoder):
    """
    Orthogonal Matching Pursuit SAE encoder.

    Uses OMP to find sparse codes: at each iteration, select the atom most correlated
    with the residual, then solve NNLS over selected atoms. Requires the decoder to be
    set via set_decoder(decoder) so the encoder can use decoder.W_dec as the dictionary.
    Encoding is non-differentiable (no_grad). Use penalty=None when using this encoder.
    """

    def __init__(self, act_size, dict_size, k=1, dropout=None, max_iter=10,
                 device='cuda', input_unit_norm=False, seed=None, concat_dim=-1, **kwargs):
        assert isinstance(k, int) and k > 0, "k must be a positive integer."
        if dropout is not None:
            assert 0.0 <= dropout <= 1.0, "Dropout must be in range [0, 1]."
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be a positive integer."
        if batched_matrix_nnls is None:
            raise ImportError("OMPEncoder requires overcomplete.optimization.batched_matrix_nnls")
        super().__init__(act_size, dict_size, device, input_unit_norm, seed, concat_dim, **kwargs)
        self.k = k
        self.dropout = dropout
        self.max_iter = max_iter
        self._decoder = None

    def set_decoder(self, decoder):
        """Set the decoder so this encoder can use its W_dec as the dictionary."""
        self._decoder = decoder

    def _encode(self, x):
        if self._decoder is None:
            raise RuntimeError("OMPEncoder requires decoder to be set. Call set_decoder(decoder) or use SAE/LightningSAE which does this automatically.")
        W = self._decoder.W_dec  # [dict_size, act_size]
        device = W.device
        dtype = W.dtype
        k = self.k
        max_iter = self.max_iter

        if self.dropout is not None and self.training:
            drop_mask = torch.bernoulli(
                (1.0 - self.dropout) * torch.ones(W.shape[0], device=device, dtype=dtype)
            )
            W = W * drop_mask.unsqueeze(1)

        x, _, _ = self.preprocess_input(x)
        x = x.to(device=device, dtype=dtype)
        batch_size = x.shape[0]
        codes = torch.zeros(batch_size, self.dict_size, device=device, dtype=dtype)
        residual = x.clone()
        selected_atoms = None

        with torch.no_grad():
            for _ in range(k):
                z_corr = residual @ W.T  # [B, dict_size]
                if selected_atoms is not None:
                    z_corr.scatter_(dim=1, index=selected_atoms, value=-torch.inf)
                _, idx = torch.topk(z_corr, k=1, dim=1)  # [B, 1]
                selected_atoms = idx if selected_atoms is None else torch.cat([selected_atoms, idx], dim=1)  # [B, num_selected]
                W_sel = W[selected_atoms]  # [B, num_selected, act_size]
                Z_init = torch.gather(codes, 1, selected_atoms)  # [B, num_selected]
                codes_sel = batched_matrix_nnls(
                    W_sel, x, max_iter=max_iter, tol=1e-5, Z_init=Z_init
                )
                codes.scatter_(dim=1, index=selected_atoms, src=codes_sel)
                residual = x - codes @ W

        return None, codes

    def train(self, mode=True):
        if not mode:
            self.dropout = None
        return super().train(mode)


class MPEncoder(SAEEncoder):
    """
    Matching Pursuit SAE encoder.

    Greedy MP: at each of k iterations, pick the atom most correlated with the
    residual, add that contribution to the codes, and subtract it from the residual.
    No NNLS (unlike OMP). Requires the decoder to be set via set_decoder(decoder).
    Encoding is non-differentiable (no_grad). Use penalty=None when using this encoder.
    """

    def __init__(self, act_size, dict_size, k=1, dropout=None,
                 device='cuda', input_unit_norm=False, seed=None, concat_dim=-1, **kwargs):
        assert isinstance(k, int) and k > 0, "k must be a positive integer."
        if dropout is not None:
            assert 0.0 <= dropout <= 1.0, "Dropout must be in range [0, 1]."
        super().__init__(act_size, dict_size, device, input_unit_norm, seed, concat_dim, **kwargs)
        self.k = k
        self.dropout = dropout
        self._decoder = None

    def set_decoder(self, decoder):
        """Set the decoder so this encoder can use its W_dec as the dictionary."""
        self._decoder = decoder

    def _encode(self, x):
        if self._decoder is None:
            raise RuntimeError(
                "MPEncoder requires decoder to be set. Call set_decoder(decoder) or use SAE/LightningSAE which does this automatically."
            )
        W = self._decoder.W_dec  # [dict_size, act_size]
        device = W.device
        dtype = W.dtype

        if self.dropout is not None and self.training:
            drop_w = torch.bernoulli(
                (1.0 - self.dropout) * torch.ones(W.shape[0], 1, device=device, dtype=dtype)
            )
            W = W * drop_w

        x, _, _ = self.preprocess_input(x)
        x = x.to(device=device, dtype=dtype)
        codes = torch.zeros(x.shape[0], self.dict_size, device=device, dtype=dtype)
        residual = x.clone()

        with torch.no_grad():
            for _ in range(self.k):
                z = residual @ W.T  # [B, dict_size]
                val, idx = torch.max(z, dim=1)  # val [B], idx [B]
                to_add = F.one_hot(idx, num_classes=self.dict_size).float() * val.unsqueeze(1)
                codes = codes + to_add
                residual = residual - to_add @ W

        return None, codes

    def train(self, mode=True):
        if not mode:
            self.dropout = None
        return super().train(mode)
