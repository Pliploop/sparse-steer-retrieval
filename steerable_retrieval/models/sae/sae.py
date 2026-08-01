import torch
import torch.nn as nn
import lightning as L
from steerable_retrieval.models.base import BaseModule
from steerable_retrieval.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Loss key -> log prefix (term first: L0/Train/..., L1/val/dataset/...)
LOSS_KEY_TO_PREFIX = {
    "l0_norm": "L0",
    "l1_loss": "L1",
    "l2_loss": "L2",
    "l1_norm": "L1_norm",
    "aux_loss": "aux_loss",
    "loss": "loss",
    "mse": "mse",
    "r2": "r2",
    "ndn": "ndn",
    "min_l2_loss": "min_L2",
    "max_l2_loss": "max_L2",
}

import logging

class SAE(BaseModule):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 sae_encoder,
                 sae_decoder,
                 ckpt_path = None,
                 freeze = False,
                 tied = False,
                 **kwargs):
        super().__init__(ckpt_path=ckpt_path, freeze=freeze)
        self.ckpt_path = ckpt_path
        self.freeze = freeze

        # Store encoder/decoder components as instance attributes
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.sae_encoder = sae_encoder
        self.sae_decoder = sae_decoder

        # Share b_dec between encoder and decoder
        if not hasattr(self.sae_decoder, 'b_dec'):
            self.sae_decoder.b_dec = self.sae_encoder.b_dec

        # OMP-style encoders need the decoder's dictionary (W_dec) at encode time
        if getattr(self.sae_encoder, 'set_decoder', None) is not None:
            self.sae_encoder.set_decoder(self.sae_decoder)

        if hasattr(self.sae_encoder, '_init_decoder_weights'):
            self.sae_encoder._init_decoder_weights(
                self.sae_decoder.W_dec
            )

        self.tied = tied

        # get the number of concepts from the output of the sae_encoder
        self.num_concepts = sae_encoder.out_channels

    def forward(self, x):
        """
        Forward pass through SAE. Expects already-encoded features [B, act_size].
        Encoding (audio_encoder / text_encoder) is done at training/validation step level.
        """
        if self.tied:
            if hasattr(self.sae_encoder, '_init_decoder_weights'):
                self.sae_encoder._init_decoder_weights(
                    self.sae_decoder.W_dec
                )
        pre, z = self.sae_encoder(x)    
        xhat = self.sae_decoder(z)
        return x, z, xhat, pre

    @torch.no_grad()
    def inference(self, x, matryoshka_k=-1):
        """
        Inference pass. Expects already-encoded features [B, act_size].
        Caller must encode with audio_encoder or text_encoder before calling.
        """
        pre, z = self.sae_encoder(x)
        xhat = self.sae_decoder(z)
        return x, z, xhat, pre

class LightningSAE(SAE, L.LightningModule):

    def __init__(self,
                audio_encoder,
                text_encoder,
                sae_encoder,
                sae_decoder,
                optimizer = None,
                scheduler = None,
                loss_fn = None,
                ckpt_path = None,
                freeze = False,
                preextracted_features = False,
                tied = False,
                data_mixing_ratio = 0.5,
                **kwargs):
        L.LightningModule.__init__(self)
        super(LightningSAE, self).__init__(
            audio_encoder=audio_encoder,
            text_encoder=text_encoder,
            sae_encoder=sae_encoder,
            sae_decoder=sae_decoder,
            ckpt_path=ckpt_path,
            freeze=freeze, 
            tied=tied,
            **kwargs
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.preextracted_features = preextracted_features
        self.reset_activations_dicts()
        self.data_mixing_ratio = data_mixing_ratio

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to only save SAE encoder/decoder weights,
        excluding audio_encoder and text_encoder.
        """
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Filter out audio_encoder and text_encoder parameters
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not (key.startswith('audio_encoder.') or key.startswith('text_encoder.')):
                filtered_state_dict[key] = value

        if hasattr(self, 'bridges'):
            filtered_state_dict['bridges'] = self.bridges
        else:
            logging.warning("No bridges found in state_dict")
        
        return filtered_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to only load SAE encoder/decoder weights.
        """
        # Restore non-parameter bridge artifacts saved by callbacks.
        # These are plain dicts, not nn.Parameters/buffers, so we need to set them manually.
        bridges = state_dict.pop('bridges', None)
        if bridges is None:
            logging.warning("No bridges found in state_dict")
        if bridges is not None:
            self.bridges = bridges

        # Filter out audio_encoder and text_encoder from the state_dict if present
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not (key.startswith('audio_encoder.') or key.startswith('text_encoder.')):
                filtered_state_dict[key] = value
        
        return super().load_state_dict(filtered_state_dict, strict=strict)

    def reset_activations_dicts(self):
        """Reset activation and embedding dictionaries for all dataloaders."""
        # Initialize empty dicts; will be populated per dataloader_idx
        self.val_activations = {}
        self.test_activations = {}
        self.val_embeddings = {}
        self.test_embeddings = {}
    
    def _get_dataset_name(self, dataloader_idx: int, mode: str) -> str:
        """Get dataset name for a dataloader from datamodule.names (same logic as callbacks)."""
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "names"):
            names = dm.names
            if isinstance(names, dict) and mode in names:
                name = names[mode].get(dataloader_idx)
                if name is not None:
                    return str(name)
        return f"dataloader_{dataloader_idx}"

    def _get_or_init_dataloader_storage(self, dataloader_idx, mode='val'):
        """Get or initialize the activation and embedding dicts for a specific dataloader_idx."""
        act_store = self.val_activations if mode == 'val' else self.test_activations
        emb_store = self.val_embeddings if mode == 'val' else self.test_embeddings
        if dataloader_idx not in act_store:
            act_store[dataloader_idx] = {'audio': [], 'text': []}
        if dataloader_idx not in emb_store:
            emb_store[dataloader_idx] = {'audio': [], 'text': []}
        return act_store[dataloader_idx], emb_store[dataloader_idx]

    def _forward_and_loss(self, encoded, loss_kw):
        """Forward pass + loss computation. Returns (loss_dict, activations, pre_penalty_activations)."""
        _, activations, reconstruction, pre_penalty_activations = self(encoded)
        loss_kw = {**loss_kw, "pre_penalty_activations": pre_penalty_activations}
        loss_dict = self.loss_fn(reconstruction, activations, encoded, **loss_kw)
        return loss_dict, activations, pre_penalty_activations

    def _log_loss_dict(self, loss_dict, stage, modality=None):
        """Log all keys in loss_dict with appropriate prefixes."""
        for key, val in loss_dict.items():
            p = LOSS_KEY_TO_PREFIX.get(key, key)
            suffix = f"/{modality}" if modality else ""
            self.log(f'{p}/{stage}{suffix}', val, prog_bar=True, on_step=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        audio, text = batch['audio'], batch['prompt']
        has_audio = audio is not None and self.data_mixing_ratio <1.0
        has_text = text is not None and self.data_mixing_ratio >0.0
        loss_kw = dict(
            num_batches_not_active=getattr(self.sae_encoder, 'num_batches_not_active', None),
            decoder=self.sae_decoder,
        )

        # Encode inputs
        encoded_audio = (self.audio_encoder(audio) if not self.preextracted_features else audio) if has_audio else None
        encoded_text = self.text_encoder(text) if has_text else None

        if has_audio and has_text:
            # Half-and-half: take half of each modality
            ratio = self.data_mixing_ratio
            n_text = int(len(text) * ratio) # if ratio = 1, only text is used
            n_audio = len(audio) - n_text # if ratio = 0, only audio is used
            encoded_audio, encoded_text = encoded_audio[:n_audio], encoded_text[:n_text]
            encoded = torch.cat([encoded_audio, encoded_text], dim=0)
            n_audio = encoded_audio.size(0)

            _, activations, reconstruction, pre_penalty_activations = self(encoded)

            # Split outputs and compute losses per modality
            def split(t): return (t[:n_audio], t[n_audio:]) if t is not None else (None, None)
            enc_a, enc_t = split(encoded)
            act_a, act_t = split(activations)
            rec_a, rec_t = split(reconstruction)
            pre_a, pre_t = split(pre_penalty_activations)
            modality_losses = {}
            if enc_a is not None and enc_a.size(0) > 0:
                modality_losses["audio"] = self.loss_fn(
                    rec_a, act_a, enc_a, **{**loss_kw, "pre_penalty_activations": pre_a}
                )
                self._log_loss_dict(modality_losses["audio"], "Train", "audio")
            if enc_t is not None and enc_t.size(0) > 0:
                modality_losses["text"] = self.loss_fn(
                    rec_t, act_t, enc_t, **{**loss_kw, "pre_penalty_activations": pre_t}
                )
                self._log_loss_dict(modality_losses["text"], "Train", "text")

            if not modality_losses:
                raise ValueError(
                    "Both audio and text slices are empty after applying data_mixing_ratio "
                    f"({self.data_mixing_ratio})."
                )

            # Log averaged total over modalities that are actually present.
            first_loss_dict = next(iter(modality_losses.values()))
            for key in first_loss_dict:
                vals = [loss_dict[key] for loss_dict in modality_losses.values() if key in loss_dict]
                if not vals:
                    continue
                p = LOSS_KEY_TO_PREFIX.get(key, key)
                self.log(
                    f'{p}/Train/total',
                    sum(vals) / len(vals),
                    prog_bar=True,
                    on_step=True,
                    sync_dist=True,
                )
            total_loss = sum(loss_dict["loss"] for loss_dict in modality_losses.values()) / len(modality_losses)
        else:
            # Single modality
            encoded = encoded_audio if has_audio else encoded_text
            modality = "audio" if has_audio else "text"
            loss_dict, _, _ = self._forward_and_loss(encoded, loss_kw)
            self._log_loss_dict(loss_dict, "Train", modality)
            total_loss = loss_dict['loss']

        self.log('loss/Train/total', total_loss, prog_bar=True, on_step=True, sync_dist=True)
        return total_loss
    
    def _eval_step(self, batch, batch_idx, dataloader_idx, stage):
        """Shared logic for validation_step and test_step."""
        audio, text = batch['audio'], batch['prompt']
        dataset_name = self._get_dataset_name(dataloader_idx, stage)
        acts, embs = self._get_or_init_dataloader_storage(dataloader_idx, mode=stage)
        loss_kw = dict(
            num_batches_not_active=getattr(self.sae_encoder, 'num_batches_not_active', None),
            decoder=self.sae_decoder,
        )
        loss_dicts = {}

        # Process each modality
        for modality, data, encoder in [
            ('audio', audio, lambda x: self.audio_encoder(x) if not self.preextracted_features else x),
            ('text', text, self.text_encoder),
        ]:
            if data is None:
                continue
            encoded = encoder(data)
            loss_dict, activations, _ = self._forward_and_loss(encoded, loss_kw)
            loss_dicts[modality] = loss_dict
            acts[modality].append(activations)
            embs[modality].append(encoded.detach())
            for key, val in loss_dict.items():
                p = LOSS_KEY_TO_PREFIX.get(key, key)
                self.log(f'{p}/{stage}/{dataset_name}/{modality}', val, sync_dist=True)

        # Compute total loss
        if not loss_dicts:
            raise ValueError(f"Both audio and text are None in {stage} batch")
        if len(loss_dicts) == 2:
            for key in loss_dicts['audio']:
                p = LOSS_KEY_TO_PREFIX.get(key, key)
                avg = (loss_dicts['audio'][key] + loss_dicts['text'][key]) / 2
                self.log(f'{p}/{stage}/{dataset_name}', avg, sync_dist=True)
            total_loss = (loss_dicts['audio']['loss'] + loss_dicts['text']['loss']) / 2
        else:
            total_loss = next(iter(loss_dicts.values()))['loss']

        self.log(f'loss/{stage}/{dataset_name}', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, batch_idx, dataloader_idx, 'test')

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.reset_activations_dicts()
    
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.reset_activations_dicts()

    # def on_before_optimizer_step(self, optimizer):
        # Ensure decoder directions stay unit-norm (TopK paper recipe)
        # if hasattr(self.sae_decoder, "make_decoder_weights_and_grad_unit_norm"):
        #     self.sae_decoder.make_decoder_weights_and_grad_unit_norm()

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            self.sae_decoder.W_dec.data = self.sae_decoder.W_dec.data / (self.sae_decoder.W_dec.data.norm(dim=-1, keepdim=True) + 1e-8)


    