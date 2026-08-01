"""Shared fixtures for the steering API tests.

All tests run on a small **untrained** synthetic BatchTopK SAE: they verify the
inversion/steering/retrieval plumbing (shapes, signs, invariants), not model quality.
"""
import hashlib

import numpy as np
import pytest
import torch

from steerable_retrieval.models.sae.decoders import SAEDecoder
from steerable_retrieval.models.sae.encoders import BatchTopKSAEEncoder
from steerable_retrieval.steer.inversion import fit_mahalanobis_prior
from steerable_retrieval.steer.loading import SteerableSAE

D, M, K, N = 64, 256, 8, 400
DEVICE = "cpu"


class StableTextEncoder:
    """Deterministic per-string embedding (stand-in for the MuQ text tower).

    Uses a stable hash so embeddings are reproducible across processes.
    """

    def __init__(self, d):
        self.d = d

    def __call__(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest(), 16) % (2**31)
            g = torch.Generator().manual_seed(seed)
            out.append(torch.randn(self.d, generator=g))
        return torch.stack(out, dim=0)


@pytest.fixture(scope="session")
def dims():
    return {"D": D, "M": M, "K": K, "N": N}


@pytest.fixture(scope="session")
def model():
    torch.manual_seed(0)
    enc = BatchTopKSAEEncoder(act_size=D, dict_size=M, top_k=K, device=DEVICE, input_unit_norm=True)
    dec = SAEDecoder(dict_size=M, act_size=D, device=DEVICE, input_unit_norm=True, output_unit_norm=True)
    dec.b_dec = enc.b_dec
    enc.eval()
    dec.eval()
    with torch.no_grad():  # prime the BatchTopK running threshold used at eval time
        enc(torch.randn(64, D))
    return SteerableSAE(enc, dec, StableTextEncoder(D))


@pytest.fixture(scope="session")
def audio(dims):
    rng = np.random.RandomState(0)
    return rng.randn(dims["N"], dims["D"]).astype(np.float32)


@pytest.fixture(scope="session")
def prior(audio):
    return fit_mahalanobis_prior(audio, device=DEVICE)
