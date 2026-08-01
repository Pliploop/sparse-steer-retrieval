"""SAE decoder classes that return only reconstruction."""

import torch
import torch.nn as nn


class SAEDecoder(nn.Module):
    """Base SAE decoder that takes activations and returns reconstruction."""

    def __init__(self, dict_size, act_size, device='cuda', 
                 input_unit_norm=False, seed=None, output_unit_norm=False):
        """
        Initialize decoder.
        
        Args:
            dict_size: Dictionary size (number of features)
            act_size: Activation size (input/output feature dimension)
            device: Device to use
            input_unit_norm: Whether to apply unit norm postprocessing
            seed: Random seed for initialization
        """
        super().__init__()
        self.dict_size = dict_size
        self.act_size = act_size
        self.device = device
        self.dtype = torch.float32  # Automatic dtype
        self.input_unit_norm = input_unit_norm
        self.output_unit_norm = output_unit_norm
        if seed is not None:
            torch.manual_seed(seed)
        
        # Decoder weights - initialize randomly, can be set from encoder later
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(dict_size, act_size)
            )
        )
        
        # b_dec will be set separately to share with encoder
        # Set it after initialization: decoder.b_dec = encoder.b_dec
        
        self.to(self.dtype).to(device)
        
        # For compatibility
        self.in_channels = dict_size
        self.out_channels = act_size

    def forward(self, activations):
        """
        Forward pass through SAE decoder.
        
        Args:
            activations: Feature activations [batch, ..., dict_size]
            
        Returns:
            reconstruction: Reconstructed features [batch, ..., act_size]
        """
        if not hasattr(self, 'b_dec'):
            raise RuntimeError("b_dec not set. This should be set automatically by SAE/LightningSAE during initialization.")
        
        reconstruction = activations @ self.W_dec + self.b_dec
        
        # Apply postprocessing if needed
        if self.input_unit_norm:
            # Note: We don't have x_mean and x_std here, so we skip postprocessing
            # This is fine since preprocessing/postprocessing are typically used together
            pass
        
        if self.output_unit_norm:
            reconstruction = reconstruction / reconstruction.norm(dim=-1, keepdim=True)
        
        return reconstruction

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Normalize decoder weights and adjust gradients."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
                -1, keepdim=True
            ) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed
