#!/usr/bin/env python3
"""
Build a concept library for MuQ (MuQ-MuLan) from caption n-grams + groundedness on precomputed audio embeddings.

Assumptions:
- You have a local folder of ~5k audio embedding .npy files (each is [D] or [1,D] or [T,D] -> we pool to [D]).
- You have a CSV on S3 with a column 'caption' containing text captions.
- You have access to MuQMuLan via `from muq_mulan import MuQMuLan` (adjust import to your project).

What it does:
1) Load captions from s3://... CSV.
2) Extract 1–3-gram candidates (stopword filtered + min docfreq).
3) Embed all candidate phrases with MuQ text tower.
4) Compute groundedness tail score against your audio embedding set.
5) Select a diverse set via greedy MMR (optional).
6) Save CSV/JSONL with concepts + stats.

Example:
python build_concept_library.py \
  --audio_dir /path/to/muq_audio_npys \
  --captions_s3 s3://my-bucket/path/captions.csv \
  --out_csv concepts.csv \
  --max_candidates 5000 \
  --final_k 250 \
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# -----------------------------
# Text encoder (given)
# -----------------------------
# IMPORTANT: adjust imports to your codebase.
# from openmuq import MuQMuLan  # <-- likely incorrect; replace with your actual import
from steerable_retrieval.models.encoders.muq import MuQTextEncoder

# -----------------------------
# S3 CSV loader
# -----------------------------
def read_captions_from_s3_csv(s3_uri: str, caption_col: str = "caption", max_rows: Optional[int] = None) -> List[str]:
    """
    Read captions from an S3 CSV using boto3 streaming.
    Requires AWS credentials in environment / config.

    s3_uri: s3://bucket/key.csv
    """
    import boto3
    from botocore.config import Config

    m = re.match(r"^s3://([^/]+)/(.+)$", s3_uri)
    if not m:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    bucket, key = m.group(1), m.group(2)

    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]

    # Stream decode bytes -> text lines
    lines = (line.decode("utf-8", errors="replace") for line in body.iter_lines())

    reader = csv.DictReader(lines)
    caps: List[str] = []
    for i, row in enumerate(reader):
        if caption_col not in row:
            raise KeyError(f"CSV missing column '{caption_col}'. Columns: {list(row.keys())}")
        cap = (row[caption_col] or "").strip()
        if cap:
            caps.append(cap)
        if max_rows is not None and (i + 1) >= max_rows:
            break
    return caps


# -----------------------------
# Caption n-gram mining
# -----------------------------
_DEFAULT_STOPWORDS = set(
    """
a an the and or but if while with without within into onto of for in on at by from as
is are was were be been being have has had do does did can could may might will would
this that these those it its it's i you we they them our your their
music song track sound sounds audio instrumental vocals voice listen listening
featuring feat ft version remix edit mix original
genre genres follows follow under includes include including characterized
scenario scenarios scene scenes depicting suitable settings overall
creates creating evokes around approximately set one main piece
tempo bpm key signature time
0 1 2 3 4 5 6 7 8 9
""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def _get_merged_stopwords() -> set:
    """Return built-in stopwords merged with NLTK English stopwords when available."""
    stopwords = set(_DEFAULT_STOPWORDS)
    try:
        from nltk.corpus import stopwords as nltk_stopwords  # type: ignore
        try:
            stopwords.update(nltk_stopwords.words("english"))
        except LookupError:
            import nltk  # type: ignore

            nltk.download("stopwords", quiet=True)
            stopwords.update(nltk_stopwords.words("english"))
    except Exception as exc:
        warnings.warn(f"NLTK stopwords unavailable, using built-in stopwords only: {exc}")
    return stopwords


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


def extract_ngrams(tokens: List[str], n_min: int = 1, n_max: int = 3) -> List[str]:
    out: List[str] = []
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        for i in range(0, L - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


def mine_ngram_candidates(
    captions: List[str],
    n_min: int = 1,
    n_max: int = 3,
    stopwords: Optional[set] = None,
    min_df: int = 5,
    max_df_frac: float = 0.30,
    max_candidates: int = 5000,
) -> List[Tuple[str, int]]:
    """
    Returns list of (ngram, df) sorted by df desc, then length desc.
    Uses document frequency: presence in a caption at least once.
    """
    if stopwords is None:
        stopwords = _get_merged_stopwords()

    df: Dict[str, int] = {}
    N = len(captions)

    for cap in captions:
        toks = [t for t in tokenize(cap) if t not in stopwords]
        if not toks:
            continue
        grams = set(extract_ngrams(toks, n_min, n_max))
        for g in grams:
            df[g] = df.get(g, 0) + 1

    # filter by df
    max_df = int(max_df_frac * N)
    items = [(g, c) for g, c in df.items() if c >= min_df and c <= max_df]

    # sort by df, then prefer longer phrases
    items.sort(key=lambda x: (x[1], len(x[0].split())), reverse=True)

    if len(items) > max_candidates:
        items = items[:max_candidates]
    return items


# -----------------------------
# Audio embedding loading
# -----------------------------
def load_audio_embeddings(audio_dir: str, max_files: Optional[int] = None) -> np.ndarray:
    """
    Loads .npy files, returns array [N, D].
    Pools if embedding has shape [T, D] by mean over T.
    """
    paths = sorted(glob.glob(os.path.join(audio_dir, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {audio_dir}")
    if max_files is not None:
        paths = paths[:max_files]

    embs: List[np.ndarray] = []
    D: Optional[int] = None
    for p in paths:
        x = np.load(p)
        x = np.asarray(x)
        if x.ndim == 2:
            # [1, D] or [T, D]
            if x.shape[0] == 1:
                x = x[0]
            else:
                x = x.mean(axis=0)
        elif x.ndim == 1:
            pass
        else:
            raise ValueError(f"Unexpected embedding shape {x.shape} in {p}")

        if D is None:
            D = int(x.shape[0])
        elif int(x.shape[0]) != D:
            raise ValueError(f"Dimension mismatch: got {x.shape[0]} vs expected {D} in {p}")

        embs.append(x.astype(np.float32))

    A = np.stack(embs, axis=0)  # [N, D]
    return A


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


# -----------------------------
# Groundedness scoring
# -----------------------------
@dataclass
class GroundScore:
    concept: str
    df: int
    mean: float
    topq_mean: float
    tail_gap: float
    gini: float
    pos_mass: float  # fraction of sims > 0


def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini for nonnegative values; for similarities, we shift to nonnegative.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    # Gini = 1 - 2 * sum_i (cum_i) / (n * cum_n) + 1/n
    return float(1.0 - (2.0 * np.sum(cum) / (n * cum[-1] + 1e-12)) + (1.0 / n))


@torch.no_grad()
def embed_text_batch(
    enc: MuQTextEncoder, texts: List[str], batch_size: int = 64, device: str = "cuda"
) -> np.ndarray:
    feats: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        t = enc(batch).to(device)
        t = t.float()
        t = torch.nn.functional.normalize(t, dim=-1)
        feats.append(t.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_groundedness(
    text_embs: np.ndarray,  # [C, D] L2-normalized
    audio_embs: np.ndarray,  # [N, D] L2-normalized
    concepts: List[str],
    dfs: List[int],
    top_q: float = 0.01,
) -> List[GroundScore]:
    """
    For each concept embedding, compute similarity distribution vs audio embeddings and derive scores.
    """
    A = audio_embs
    C = text_embs
    assert C.shape[0] == len(concepts) == len(dfs)

    scores: List[GroundScore] = []

    # Compute in chunks to avoid huge memory if C is large.
    # sims = A @ C.T -> [N, C]
    # With N=5000, C up to 5000, this is 25M ~ 100MB float32; still OK, but chunk anyway.
    N = A.shape[0]
    qk = max(1, int(math.ceil(top_q * N)))

    chunk = 512
    for j0 in range(0, C.shape[0], chunk):
        j1 = min(C.shape[0], j0 + chunk)
        Cc = C[j0:j1]  # [c, D]
        sims = A @ Cc.T  # [N, c]

        for jj in range(j1 - j0):
            s = sims[:, jj].astype(np.float64)
            mean = float(s.mean())
            # top-q mean
            top_idx = np.argpartition(s, -qk)[-qk:]
            topq_mean = float(s[top_idx].mean())
            tail_gap = topq_mean - mean

            # groundedness also reflected by concentration: compute gini on shifted sims
            s_shift = s - s.min()
            gini = gini_coefficient(s_shift)

            pos_mass = float((s > 0).mean())

            scores.append(
                GroundScore(
                    concept=concepts[j0 + jj],
                    df=dfs[j0 + jj],
                    mean=mean,
                    topq_mean=topq_mean,
                    tail_gap=tail_gap,
                    gini=gini,
                    pos_mass=pos_mass,
                )
            )
    return scores


# -----------------------------
# Diversity selection (MMR)
# -----------------------------
def mmr_select(
    concepts: List[str],
    text_embs: np.ndarray,  # [C, D] L2-normalized
    base_score: np.ndarray,  # [C]
    k: int,
    lambda_mmr: float = 0.7,
) -> List[int]:
    """
    Greedy Maximal Marginal Relevance selection.
    """
    C = text_embs
    selected: List[int] = []
    remaining = set(range(len(concepts)))

    # Precompute similarity matrix chunk-wise if needed; for C <= 5000 OK but keep simple:
    # We'll compute max similarity to selected on the fly.
    while len(selected) < k and remaining:
        best_i = None
        best_val = -1e18
        for i in list(remaining):
            if not selected:
                val = float(base_score[i])
            else:
                # max similarity to selected
                sims = C[selected] @ C[i]
                max_sim = float(np.max(sims))
                val = lambda_mmr * float(base_score[i]) - (1.0 - lambda_mmr) * max_sim
            if val > best_val:
                best_val = val
                best_i = i
        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=str, required=True)
    ap.add_argument("--captions_s3", type=str, required=True)
    ap.add_argument("--caption_col", type=str, default="caption")
    ap.add_argument("--max_captions", type=int, default=10000)

    ap.add_argument("--n_min", type=int, default=1)
    ap.add_argument("--n_max", type=int, default=3)
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--max_df_frac", type=float, default=0.30)
    ap.add_argument("--max_candidates", type=int, default=5000)

    ap.add_argument("--model_name", type=str, default="OpenMuQ/MuQ-MuLan-large")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--text_batch_size", type=int, default=16)

    ap.add_argument("--top_q", type=float, default=0.01)
    ap.add_argument("--final_k", type=int, default=20)
    ap.add_argument("--use_mmr", action="store_true")
    ap.add_argument("--lambda_mmr", type=float, default=0.7)

    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, default=None)

    args = ap.parse_args()

    print(f"[1/6] Loading audio embeddings from {args.audio_dir} ...")
    audio = load_audio_embeddings(args.audio_dir)
    audio = l2_normalize_np(audio)
    print(f"Loaded audio embeddings: {audio.shape}")

    print(f"[2/6] Reading captions from {args.captions_s3} ...")
    captions = read_captions_from_s3_csv(args.captions_s3, caption_col=args.caption_col, max_rows=args.max_captions)
    print(f"Loaded captions: {len(captions)}")

    print("[3/6] Mining n-gram candidates ...")
    cand = mine_ngram_candidates(
        captions=captions,
        n_min=args.n_min,
        n_max=args.n_max,
        min_df=args.min_df,
        max_df_frac=args.max_df_frac,
        max_candidates=args.max_candidates,
    )
    concepts = [g for g, _ in cand]
    dfs = [df for _, df in cand]
    print(f"Candidates after filtering: {len(concepts)}")

    print("[4/6] Embedding candidate concepts with MuQ text encoder ...")
    enc = MuQTextEncoder(model_name=args.model_name, device=args.device, freeze=True)
    text_embs = embed_text_batch(enc, concepts, batch_size=args.text_batch_size, device=args.device)
    text_embs = l2_normalize_np(text_embs)
    print(f"Text embeddings: {text_embs.shape}")

    print("[5/6] Computing groundedness scores ...")
    scores = compute_groundedness(
        text_embs=text_embs,
        audio_embs=audio,
        concepts=concepts,
        dfs=dfs,
        top_q=args.top_q,
    )

    # Base score for ranking: tail_gap (primary) * log(df+1) (secondary) + small gini
    base = np.array([s.tail_gap for s in scores], dtype=np.float64)
    dfw = np.log(np.array([s.df for s in scores], dtype=np.float64) + 1.0)
    gini = np.array([s.gini for s in scores], dtype=np.float64)
    base_score = base * dfw + 0.05 * gini

    # Sort by base_score
    order = np.argsort(-base_score)
    if args.use_mmr:
        # Run MMR on top pool to keep it fast
        pool = order[: min(len(order), max(args.final_k * 10, 500))]
        pool_concepts = [concepts[i] for i in pool]
        pool_embs = text_embs[pool]
        pool_base = base_score[pool]
        sel_pool_idx = mmr_select(pool_concepts, pool_embs, pool_base, k=args.final_k, lambda_mmr=args.lambda_mmr)
        selected = [pool[i] for i in sel_pool_idx]
    else:
        selected = order[: args.final_k].tolist()

    print(f"[6/6] Writing outputs: top-{len(selected)} concepts -> {args.out_csv}")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    # also write the concepts to a text file
    with open(args.out_csv.replace(".csv", ".txt"), "w", encoding="utf-8") as f:
        for i in selected:
            f.write(scores[i].concept + "\n")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "df", "mean_sim", "topq_mean_sim", "tail_gap", "gini", "pos_mass", "base_score"])
        for i in selected:
            s = scores[i]
            w.writerow([s.concept, s.df, s.mean, s.topq_mean, s.tail_gap, s.gini, s.pos_mass, float(base_score[i])])

    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for i in selected:
                s = scores[i]
                f.write(
                    json.dumps(
                        {
                            "concept": s.concept,
                            "df": s.df,
                            "mean_sim": s.mean,
                            "topq_mean_sim": s.topq_mean,
                            "tail_gap": s.tail_gap,
                            "gini": s.gini,
                            "pos_mass": s.pos_mass,
                            "base_score": float(base_score[i]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print("Done.")


if __name__ == "__main__":
    main()
