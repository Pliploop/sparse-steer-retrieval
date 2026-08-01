"""Fit and ship the default Mahalanobis manifold prior.

The prior is a fixed distributional constant of the audio manifold, fit once on a
sample of pre-extracted audio embeddings and packaged with the library so the
:class:`~steerable_retrieval.steer.slider.Slider` works with no data from the caller.
It is deliberately independent of any retrieval corpus.

CLI (produces the packaged asset):

    python -m steerable_retrieval.steer.prior_fit \
        --manifest /path/to/embedding_lookup_manifest.csv \
        --out steerable_retrieval/assets/muq_mulan_music4all_prior.npz \
        --n-samples 20000
"""

from __future__ import annotations

import argparse
import csv
import random
from typing import Optional

import numpy as np

from steerable_retrieval.steer.inversion import MahalanobisPrior, fit_mahalanobis_prior


def load_embeddings_from_manifest(
    manifest_csv: str,
    *,
    n_samples: int = 20000,
    split: Optional[str] = "train",
    seed: int = 0,
    path_col: str = "audio_embedding_path",
    status_col: str = "audio_embedding_status",
) -> np.ndarray:
    """Sample ``n_samples`` audio-embedding ``.npy`` paths from a manifest and stack them."""
    paths = []
    with open(manifest_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if split and row.get("split") != split:
                continue
            if status_col in row and row[status_col] not in ("ok", "", None):
                continue
            p = row.get(path_col)
            if p:
                paths.append(p)

    rng = random.Random(seed)
    if n_samples and len(paths) > n_samples:
        paths = rng.sample(paths, n_samples)

    embs = []
    for p in paths:
        try:
            embs.append(np.asarray(np.load(p), dtype=np.float32).reshape(-1))
        except Exception:
            continue
    if not embs:
        raise RuntimeError(f"No embeddings loaded from {manifest_csv}")
    return np.stack(embs, axis=0)


def fit_and_save(
    manifest_csv: str,
    out_path: str,
    *,
    n_samples: int = 20000,
    split: Optional[str] = "train",
    model: str = "OpenMuQ/MuQ-MuLan-large",
) -> MahalanobisPrior:
    X = load_embeddings_from_manifest(manifest_csv, n_samples=n_samples, split=split)
    prior = fit_mahalanobis_prior(X)
    prior.save(
        out_path,
        meta={"model": model, "n_samples": int(X.shape[0]), "dim": int(X.shape[1]), "split": split, "source": "music4all"},
    )
    print(f"Fit prior on {X.shape[0]} embeddings (dim {X.shape[1]}) -> {out_path}")
    return prior


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="embedding manifest CSV")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="OpenMuQ/MuQ-MuLan-large")
    args = ap.parse_args()
    fit_and_save(args.manifest, args.out, n_samples=args.n_samples, split=args.split, model=args.model)


if __name__ == "__main__":
    main()
