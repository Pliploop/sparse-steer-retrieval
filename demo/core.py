"""Shared engine for the demo Space and website example generation.

Loads a trained BatchTopK SAE + MuQ text tower and a music4all corpus, then steers a
seed track toward free-form concepts and retrieves nearest neighbours. The SAE,
inversion, edit masks, and retrieval corpus stay on CPU; ZeroGPU is used only while
embedding a new text concept for slider instantiation.

Configuration (env):
  SSR_CHECKPOINT   local .ckpt path, or a HF model repo id (default: local L0=20 run)
  SSR_L0           subfolder in the model repo to load, e.g. "L0-20" (Hub repos only)
  SSR_CORPUS_REPO  HF dataset repo id holding corpus.npz + meta.json (overrides local dir)
  SSR_CORPUS_DIR   local corpus dir (default: demo/corpus)
  SSR_DEVICE       "cuda" / "cpu" (auto-detected)
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from steerable_retrieval.steer import Slider
from steerable_retrieval.steer.loading import load_steerable_sae
from steerable_retrieval.steer.steering import decode_normalized

HERE = os.path.dirname(__file__)
CORPUS_DIR = os.environ.get("SSR_CORPUS_DIR", os.path.join(HERE, "corpus"))
CORPUS_REPO = os.environ.get("SSR_CORPUS_REPO")  # HF dataset repo id, if hosted
CHECKPOINT = os.environ.get("SSR_CHECKPOINT", os.path.join(HERE, os.pardir, "logs/xps/66a0caf9/checkpoints/last.ckpt"))
L0_SUBFOLDER = os.environ.get("SSR_L0")  # e.g. "L0-20" when CHECKPOINT is a Hub repo
CONFIG = os.environ.get("SSR_CONFIG")
INVERSION_METHOD = os.environ.get("SSR_INVERSION_METHOD", "fista")


def device() -> str:
    return os.environ.get("SSR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def load_corpus():
    """Return (embeddings[np], track_ids[list], meta[dict]). Fetches from the HF dataset
    repo if SSR_CORPUS_REPO is set, else reads the local corpus dir. CPU only."""
    if CORPUS_REPO:
        from huggingface_hub import hf_hub_download

        npz_path = hf_hub_download(CORPUS_REPO, filename="corpus.npz", repo_type="dataset")
        meta_path = hf_hub_download(CORPUS_REPO, filename="meta.json", repo_type="dataset")
    else:
        npz_path = os.path.join(CORPUS_DIR, "corpus.npz")
        meta_path = os.path.join(CORPUS_DIR, "meta.json")
    npz = np.load(npz_path, allow_pickle=True)
    with open(meta_path) as fh:
        meta = json.load(fh)
    return npz["embeddings"].astype(np.float32), [str(t) for t in npz["track_ids"].tolist()], meta


class DemoEngine:
    def __init__(self, corpus=None, dev: str | None = None):
        # Keep SAE inversion, sparse edits, and retrieval on CPU. ZeroGPU is only
        # needed while embedding new text concepts with MuQ.
        self.device = "cpu"
        self.text_device = dev or device()
        embs, ids, meta = corpus if corpus is not None else load_corpus()
        self.embeddings = torch.from_numpy(embs).float()
        self.gallery = F.normalize(self.embeddings, dim=-1)
        self.track_ids = ids
        self.id_to_idx = {t: i for i, t in enumerate(ids)}
        self.meta = meta
        self.model = load_steerable_sae(CHECKPOINT, device="cpu", subfolder=L0_SUBFOLDER, config_path=CONFIG)
        self._slider_cache: dict[str, Slider] = {}
        self._text_encoder_device = "cpu"

    def _slider_key(self, concept: str) -> str:
        return " ".join(concept.strip().lower().split())

    def has_slider(self, concept: str) -> bool:
        return self._slider_key(concept) in self._slider_cache

    def _set_text_encoder_device(self, target: str) -> None:
        if self._text_encoder_device == target:
            return
        text_encoder = self.model.text_encoder
        if text_encoder is None:
            return
        text_encoder.to(target)
        if hasattr(text_encoder, "device"):
            text_encoder.device = target
        self._text_encoder_device = target

    def _slider(self, concept: str) -> Slider:
        key = self._slider_key(concept)
        if key not in self._slider_cache:
            target = self.text_device if str(self.text_device).startswith("cuda") and torch.cuda.is_available() else "cpu"
            self._set_text_encoder_device(target)
            try:
                self._slider_cache[key] = Slider(concept.strip(), model=self.model, method=INVERSION_METHOD, device="cpu")
            finally:
                # The cached slider stores its text embedding/mask; live edits no
                # longer need the text tower or any GPU memory.
                self._set_text_encoder_device("cpu")
        return self._slider_cache[key]

    def track_meta(self, track_id: str) -> dict:
        m = dict(self.meta.get(track_id, {}))
        m["track_id"] = track_id
        return m

    def seed_embedding(self, track_id: str) -> torch.Tensor:
        return self.embeddings[self.id_to_idx[track_id]]

    def _retrieve_dense(self, query: torch.Tensor, k: int, *, exclude_idx: int | None = None):
        query = F.normalize(query.reshape(-1), dim=0)
        sims = self.gallery @ query
        if exclude_idx is not None:
            sims[exclude_idx] = -1e9
        n_avail = self.gallery.shape[0] - (1 if exclude_idx is not None else 0)
        vals, idx = torch.topk(sims, k=min(int(k), int(n_avail)), largest=True, sorted=True)
        return idx.detach().cpu(), vals.detach().cpu()

    def steer_and_retrieve(self, seed_track_id: str, concept: str, alpha: float = 1.0, k: int = 8) -> list[dict]:
        slider = self._slider(concept)
        z = self.seed_embedding(seed_track_id)
        seed_idx = self.id_to_idx[seed_track_id]
        edited = slider.steer(z, alpha=alpha)
        idx, scores = self._retrieve_dense(edited, k, exclude_idx=seed_idx)
        return self._format_results(idx, scores)

    def multi_steer_and_retrieve(self, seed_track_id: str, sliders: list[tuple[str, float]], k: int = 8) -> list[dict]:
        """Apply several concept sliders to one seed query, then retrieve once.

        Each concept is inverted/cached independently, but the edit itself is additive
        in sparse SAE space: encode the seed once, add every active slider mask scaled
        by alpha, decode once, and search from the combined query.
        """
        z = self.seed_embedding(seed_track_id).reshape(1, -1)
        seed_idx = self.id_to_idx[seed_track_id]
        slider_masks = []
        for concept, alpha in sliders:
            concept = concept.strip()
            alpha = float(alpha)
            if not concept or abs(alpha) < 1e-6:
                continue
            slider_masks.append((self._slider(concept).mask, alpha))

        with torch.inference_mode():
            _, sparse, _, _ = self.model.inference(z)
            edited_sparse = sparse.clone()
            for mask, alpha in slider_masks:
                mask = mask.to(device=self.device, dtype=edited_sparse.dtype)
                edited_sparse = edited_sparse + alpha * mask.reshape(1, -1)
            edited_sparse = edited_sparse.clamp_min(0.0)
            edited = decode_normalized(self.model, edited_sparse)
        idx, scores = self._retrieve_dense(edited, k, exclude_idx=seed_idx)
        return self._format_results(idx, scores)

    def _format_results(self, idx, scores) -> list[dict]:
        out = []
        for i, s in zip(idx.tolist(), scores.tolist()):
            m = self.track_meta(self.track_ids[i])
            m["affinity"] = float(s)
            out.append(m)
        return out


_ENGINE: DemoEngine | None = None


def get_engine(corpus=None) -> DemoEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DemoEngine(corpus=corpus)
    return _ENGINE
