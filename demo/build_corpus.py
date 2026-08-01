"""Build a compact retrieval corpus for the demo Space / example generation.

Reads the music4all embedding manifest (one 512-d audio embedding per clip) and the
track metadata (title / artist / Spotify id / genre), samples up to N distinct tracks
that have both an embedding and a Spotify id, and writes:

    demo/corpus/corpus.npz    embeddings [N, 512] float32 + track_ids [N]
    demo/corpus/meta.json     { track_id: {title, artist, spotify, genre} }

Usage:
    python demo/build_corpus.py --n 20000
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

MANIFEST = "/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1/embeddings/embedding_lookup_manifest.csv"
TRACKS = "/gpfs/scratch/acw749/datasets/music4all_instruct/metadata/music4all_tracks.jsonl"
OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus")

csv.field_size_limit(sys.maxsize)


def load_track_meta(path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            mi = r.get("musicinfo", {}) or {}
            sp = mi.get("spotify_id")
            if not sp:
                continue
            genres = (mi.get("tags", {}) or {}).get("genres", []) or []
            meta[r["id"]] = {
                "title": r.get("name", "") or "",
                "artist": r.get("artist_name", "") or "",
                "spotify": sp,
                "genre": genres[0] if genres else "",
            }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="max distinct tracks to keep")
    args = ap.parse_args()

    print(f"loading track metadata from {TRACKS} ...")
    meta = load_track_meta(TRACKS)
    print(f"  {len(meta)} tracks with a Spotify id")

    print(f"scanning manifest {MANIFEST} ...")
    embs, ids, kept = [], [], set()
    with open(MANIFEST) as fh:
        r = csv.DictReader(fh)
        for row in r:
            if row.get("audio_embedding_status") != "ok":
                continue
            tid = row["track_id"]
            if tid in kept or tid not in meta:
                continue
            path = row["audio_embedding_path"]
            try:
                e = np.load(path).astype(np.float32).reshape(-1)
            except Exception:
                continue
            if e.shape[0] != 512:
                continue
            embs.append(e)
            ids.append(tid)
            kept.add(tid)
            if len(kept) % 2000 == 0:
                print(f"  {len(kept)} kept")
            if len(kept) >= args.n:
                break

    X = np.stack(embs).astype(np.float32)
    ids = np.array(ids)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "corpus.npz"), embeddings=X, track_ids=ids)
    sub_meta = {tid: meta[tid] for tid in ids.tolist()}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(sub_meta, fh)
    print(f"wrote {X.shape[0]} embeddings {X.shape} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
