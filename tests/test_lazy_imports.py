import subprocess
import sys


def test_sae_encoder_import_does_not_require_lightning():
    script = r"""
import builtins

orig_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "lightning" or name.startswith("lightning."):
        raise ModuleNotFoundError("No module named 'lightning'")
    return orig_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

from hydra.utils import get_class

cls = get_class("steerable_retrieval.models.sae.encoders.BatchTopKSAEEncoder")
assert cls.__name__ == "BatchTopKSAEEncoder"
"""
    subprocess.run([sys.executable, "-c", script], check=True)
