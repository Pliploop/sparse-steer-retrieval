"""Sparse Autoencoder (SAE) implementations."""

from steerable_retrieval.models.sae.encoders import (
    SAEEncoder,
    VanillaSAEEncoder,
    TopKSAEEncoder,
    BatchTopKSAEEncoder,
    MatryoshkaBatchTopKSAEEncoder,
    JumpReLUSAEEncoder,
    OMPEncoder,
    MPEncoder,
)
from steerable_retrieval.models.sae.decoders import SAEDecoder
from steerable_retrieval.models.sae.penalties import (
    VanillaPenalty,
    TopKPenalty,
    BatchTopKPenalty,
    MatryoshkaBatchTopKPenalty,
    L1Penalty,
    JumpReLU,
    StepFunction,
    RectangleFunction,
)


def __getattr__(name):
    """Load Lightning-backed SAE classes only when explicitly requested.

    Inference code often imports encoder/decoder targets through Hydra, which first
    imports this package. Keeping ``SAE`` and ``LightningSAE`` lazy avoids requiring
    the training stack, especially ``lightning``, in lightweight demo runtimes.
    """
    if name in {"SAE", "LightningSAE"}:
        from steerable_retrieval.models.sae.sae import SAE, LightningSAE

        return {"SAE": SAE, "LightningSAE": LightningSAE}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'SAE',
    'LightningSAE',
    'SAEEncoder',
    'VanillaSAEEncoder',
    'TopKSAEEncoder',
    'BatchTopKSAEEncoder',
    'MatryoshkaBatchTopKSAEEncoder',
    'JumpReLUSAEEncoder',
    'OMPEncoder',
    'MPEncoder',
    'SAEDecoder',
    'VanillaPenalty',
    'TopKPenalty',
    'BatchTopKPenalty',
    'MatryoshkaBatchTopKPenalty',
    'L1Penalty',
    'JumpReLU',
    'StepFunction',
    'RectangleFunction',
]
