"""MuQ encoder wrappers for audio and text."""

import torch
import torch.nn as nn
from muq import MuQMuLan
from steerable_retrieval.models.base import BaseModule


class MuQAudioEncoder(BaseModule):
    """MuQ audio encoder wrapper that returns dense features after projection head."""

    def __init__(self, model_name="OpenMuQ/MuQ-MuLan-large", device="cuda", freeze=True, ckpt_path=None, sampling_rate=24000, **kwargs):
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.model_name = model_name
        self.device = device
        self.freeze = freeze

        # Load MuQ model
        self.model = MuQMuLan.from_pretrained(model_name)
        self.model = self.model.to(device).eval()

        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.sampling_rate = sampling_rate

    def forward(self, audio):
        """
        Forward pass through MuQ audio encoder.
        
        Args:
            audio: Audio input. Can be:
                - Raw audio tensor [batch, samples] or [samples]
                - Numpy array
                - Already processed audio
        
        Returns:
            audio_features: Dense audio features [batch, feature_dim]
        """
        # Ensure audio is a tensor on the correct device
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        
        # Ensure audio is on the correct device
        audio = audio.to(self.device)
        
        # Handle batched audio tensors from DataLoader
        # DataLoader stacks tensors, so we get [batch, 1, samples] or [batch, samples]
        if audio.dim() == 3:
            # [batch, 1, samples] -> squeeze middle dimension
            audio = audio.squeeze(1)
        elif audio.dim() == 2:
            # [batch, samples] - already correct
            pass
        elif audio.dim() == 1:
            # [samples] - single sample, add batch dim
            audio = audio.unsqueeze(0)
        
        # Get dense audio features (after projection head)
        with torch.set_grad_enabled(not self.freeze):
            out = self.model(wavs=audio)
        # MuQ/HuggingFace models may return BaseModelOutputWithPooling; return tensor [B, D]
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
            return h[:, 0] if h.dim() == 3 else h
        raise TypeError(f"MuQ model returned {type(out).__name__}; expected tensor or pooler_output/last_hidden_state.")


class MuQTextEncoder(BaseModule):
    """MuQ text encoder wrapper that returns dense features after projection head."""

    def __init__(self, model_name="OpenMuQ/MuQ-MuLan-large", device="cuda", freeze=True, ckpt_path=None, **kwargs):
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.model_name = model_name
        self.device = device
        self.freeze = freeze

        # Load MuQ model
        self.model = MuQMuLan.from_pretrained(model_name)
        self.model = self.model.to(device).eval()

        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text):
        """
        Forward pass through MuQ text encoder.
        
        Args:
            text: Text input. Can be:
                - String
                - List of strings
        
        Returns:
            text_features: Dense text features [batch, feature_dim]
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        
        # Get dense text features (after projection head)
        with torch.set_grad_enabled(not self.freeze):
            out = self.model(texts=text)
        # MuQ/HuggingFace models may return BaseModelOutputWithPooling; return tensor [B, D]
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
            return h[:, 0] if h.dim() == 3 else h
        raise TypeError(f"MuQ model returned {type(out).__name__}; expected tensor or pooler_output/last_hidden_state.")
