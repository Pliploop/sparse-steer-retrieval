"""CLAP encoder wrappers for audio and text."""

import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor
from steerable_retrieval.models.base import BaseModule
import numpy as np

def _to_embedding_tensor(out):
    """Return [batch, feature_dim] tensor from model output (tensor or HuggingFace output)."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        h = out.last_hidden_state
        return h[:, 0] if h.dim() == 3 else h
    raise TypeError(
        f"Model returned {type(out).__name__}; expected tensor or pooler_output/last_hidden_state."
    )

import logging


class CLAPAudioEncoder(BaseModule):
    """CLAP audio encoder wrapper that returns dense features after projection head."""

    def __init__(self, model_name="laion/larger_clap_music", device="cuda", freeze=True, ckpt_path=None, sampling_rate=48000, **kwargs):
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.model_name = model_name
        self.freeze = freeze
        self.sampling_rate = sampling_rate

        # Load CLAP model and processor
        self.model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)
        # Set sampling rate to suppress warning
        if hasattr(self.processor, 'feature_extractor'):
            self.processor.feature_extractor.sampling_rate = sampling_rate

        # Set to eval mode and freeze if requested
        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        

    def forward(self, audio):
        """
        Forward pass through CLAP audio encoder.
        
        Args:
            audio: Audio input. Can be:
                - Raw audio array (numpy array or torch tensor)
                - Dictionary with 'array' key (from datasets)
                - Already processed audio tensor
        
        Returns:
            audio_features: Dense audio features [batch, feature_dim]
        """
        # Process audio if needed
        device = next(self.model.parameters()).device
        if isinstance(audio, dict) and 'array' in audio:
            # Handle dataset format
            audio_array = audio['array']
            inputs = self.processor(audio=audio_array, return_tensors="pt", sampling_rate=self.sampling_rate).to(device)
        elif isinstance(audio, torch.Tensor):
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
            
            # Convert to list of numpy arrays for CLAP processor
            # CLAP processor expects list of 1D arrays
            audio_list = [a.cpu().numpy() for a in audio]
            inputs = self.processor(audio=audio_list, return_tensors="pt", sampling_rate=self.sampling_rate).to(device)
        else:
            # Assume it's a numpy array or list
            inputs = self.processor(audio=audio, return_tensors="pt", sampling_rate=self.sampling_rate).to(device)

        
        # make sure the inputs are on the correct device
        for key, value in inputs.items():
            # if numpy array, convert to tensor
            if isinstance(value, np.ndarray):
                value = torch.tensor(value).to(device)
                inputs[key] = value

        # Get dense audio features (after projection head)
        if self.freeze:
            with torch.no_grad():
                out = self.model.get_audio_features(**inputs)
        else:
            out = self.model.get_audio_features(**inputs)
        return _to_embedding_tensor(out)


class CLAPTextEncoder(BaseModule):
    """CLAP text encoder wrapper that returns dense features after projection head."""

    def __init__(self, model_name="laion/larger_clap_music", device="cuda", freeze=True, ckpt_path=None, **kwargs):
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.model_name = model_name
        self.device = device
        self.freeze = freeze

        # Load CLAP model and processor
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)

        # Set to eval mode and freeze if requested
        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text):
        """
        Forward pass through CLAP text encoder.
        
        Args:
            text: Text input. Can be:
                - String
                - List of strings
                - Already processed text
        
        Returns:
            text_features: Dense text features [batch, feature_dim]
        """
        # Process text if needed
        if isinstance(text, str):
            text = [text]

        device = next(self.model.parameters()).device
        
        # Process text through CLAP processor
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # make sure the inputs are on the correct device
        for key, value in inputs.items():
            # if numpy array, convert to tensor
            if isinstance(value, np.ndarray):
                value = torch.tensor(value).to(device)
                inputs[key] = value
                

        # Get dense text features (after projection head)
        with torch.set_grad_enabled(not self.freeze):
            out = self.model.get_text_features(**inputs)
        return _to_embedding_tensor(out)
