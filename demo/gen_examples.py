"""Generate website examples from the trained model + corpus.

For each *regenerated* concept we pick a seed that genuinely lacks the concept (low
alignment band), choose alpha at the PEAK of the steered query's concept similarity
(going stronger reverses the edit), and retrieve — de-duplicating so the seed's own
artist never reappears and each result is a distinct artist. Concepts in KEEP_IDS are
FROZEN: their prior seed/alpha/results/curve are copied verbatim (Adam inversion is
non-deterministic, so re-running would lose them); we only de-dupe their frozen results.
ALL concepts are emitted, sorted by strength, to be trimmed by hand.

    SSR_DEVICE=cuda SSR_CHECKPOINT=<ckpt> python demo/gen_examples.py
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from core import get_engine

OUT = os.path.join(os.path.dirname(__file__), os.pardir, "website", "src", "data", "examples.js")

# Concepts frozen verbatim from the current examples.js (only their results are de-duped).
KEEP_IDS = {"mellow", "dark", "metal", "groovy"}

POOL = [
    ("punchy", "punchy", "#E23B34"),
    ("somber", "somber and melancholic", "#2E6FD6"),
    ("dreamy", "dreamy and ethereal", "#A855F7"),
    ("aggressive", "aggressive and intense", "#DC2626"),
    ("mellow", "mellow and relaxed", "#14B8A6"),
    ("energetic", "energetic and upbeat", "#F59E0B"),
    ("dark", "dark and brooding", "#6366F1"),
    ("uplifting", "uplifting and euphoric", "#EC4F9E"),
    ("groovy", "groovy and funky", "#FB8B24"),
    ("epic", "epic and cinematic", "#8B5CF6"),
    ("warm", "warm and intimate", "#B45309"),
    ("gritty", "gritty and raw", "#0EA5E9"),
    ("piano", "piano", "#7B3FF2"),
    ("distorted-guitar", "distorted electric guitar", "#E11D48"),
    ("acoustic-guitar", "acoustic guitar", "#1FA347"),
    ("female-vocals", "female vocals", "#DB2777"),
    ("hip-hop", "hip hop beats", "#F97316"),
    ("metal", "heavy metal", "#7F1D1D"),
]

ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]  # gentle; peak selection guards against reversal
N_SEED_CANDIDATES = 48
SEED_BAND = (0.0, 0.30)      # draw seeds from the LOW-alignment band (lack the concept)
SEED_ALIGN_MAX = float(os.environ.get("SSR_SEED_ALIGN_MAX", "0.05"))  # seed must be <= this
K = 6
SCORE_TOP = 3
CURVE_N = 21
MODEL_TAG = os.environ.get("SSR_MODEL_TAG", "BatchTopK SAE · MuQ-MuLan · music4all · L0=20")
RNG = torch.Generator().manual_seed(1)
# If set (comma-separated ids), regenerate ONLY these and freeze every other concept from
# the existing examples.js. Otherwise freeze KEEP_IDS and regenerate the rest.
REGEN_ONLY = {x for x in os.environ.get("SSR_REGEN_ONLY", "").split(",") if x}


def concept_alignment(engine, text_emb):
    return F.cosine_similarity(F.normalize(text_emb, dim=-1), F.normalize(engine.embeddings, dim=-1), dim=-1)


def good_meta(engine, tid):
    m = engine.meta.get(tid, {})
    return bool(m.get("title") and m.get("artist") and m.get("spotify"))


def te_and_align(engine, concept):
    with torch.inference_mode():
        te = engine.model.text_encoder([concept]).to(engine.device)
    return F.normalize(te.flatten(), dim=0), concept_alignment(engine, te)


def query_sims(engine, slider, seed_id, te_norm, alphas):
    """Steered query's concept similarity at each alpha."""
    z = engine.seed_embedding(seed_id)
    out = []
    with torch.inference_mode():
        for a in alphas:
            e = slider.steer(z, alpha=a).flatten()
            out.append(float(torch.dot(F.normalize(e, dim=0), te_norm)))
    return out


def peak_alpha(engine, slider, seed_id, te_norm):
    """Alpha maximising the steered query's concept similarity — never past the peak."""
    sims = query_sims(engine, slider, seed_id, te_norm, ALPHAS)
    return ALPHAS[max(range(len(ALPHAS)), key=lambda i: sims[i])]


def retrieve_dedup(engine, slider, seed_id, alpha, k=K):
    """Steer + retrieve, dropping the seed's own artist and keeping one track per artist."""
    z = engine.seed_embedding(seed_id)
    si = engine.id_to_idx[seed_id]
    seed_artist = engine.meta.get(seed_id, {}).get("artist", "").strip().lower()
    idx, scores = slider.retrieve(z, engine.embeddings, alpha=alpha, k=k * 8, exclude_idx=si)
    out, seen_artist, seen_track = [], set(), set()
    for i, s in zip(idx.tolist(), scores.tolist()):
        tid = engine.track_ids[i]
        if not good_meta(engine, tid):
            continue
        m = engine.meta[tid]
        a, t = m["artist"].strip().lower(), m["title"].strip().lower()
        if a == seed_artist or a in seen_artist or (a, t) in seen_track:
            continue
        seen_artist.add(a)
        seen_track.add((a, t))
        mm = engine.track_meta(tid)
        mm["affinity"] = float(s)
        out.append(mm)
        if len(out) >= k:
            break
    return out


def score_curve(engine, slider, seed_id, te_norm, alpha_max):
    z = engine.seed_embedding(seed_id)
    curve = []
    with torch.inference_mode():
        for a in torch.linspace(0.0, alpha_max, CURVE_N).tolist():
            e = slider.steer(z, alpha=a).flatten()
            curve.append(round(float(torch.dot(F.normalize(e, dim=0), te_norm)), 4))
    return curve


def ret_align_of(engine, align, results):
    top = results[:SCORE_TOP]
    if not top:
        return -1.0
    return sum(align[engine.id_to_idx[r["track_id"]]].item() for r in top) / len(top)


def build_example(engine, cid, concept, accent, seed_id, alpha, align, te_norm):
    slider = engine._slider(concept)
    results = retrieve_dedup(engine, slider, seed_id, alpha)
    curve = score_curve(engine, slider, seed_id, te_norm, alpha)
    seed = engine.track_meta(seed_id)
    ex = {
        "ret_align": round(ret_align_of(engine, align, results), 4),
        "id": cid,
        "concept": concept,
        "accent": accent,
        "alpha": alpha,
        "seed": {k: seed[k] for k in ("title", "artist", "genre", "spotify")},
        "results": [{k: r[k] for k in ("title", "artist", "genre", "affinity", "spotify")} for r in results],
        "scores": curve,
    }
    return ex, float(align[engine.id_to_idx[seed_id]])


def choose_seed(engine, concept, align, te_norm):
    """Best low-alignment seed; alpha at its steered-query peak; scored by retrieved alignment."""
    order = torch.argsort(align)
    n = len(order)
    band = order[int(SEED_BAND[0] * n): int(SEED_BAND[1] * n)]
    pick = band[torch.randperm(len(band), generator=RNG)[: N_SEED_CANDIDATES * 2]]
    slider = engine._slider(concept)
    best, tried = None, 0
    for si in pick.tolist():
        tid = engine.track_ids[si]
        if not good_meta(engine, tid) or align[si].item() > SEED_ALIGN_MAX:
            continue  # seed must genuinely lack the concept
        tried += 1
        alpha = peak_alpha(engine, slider, tid, te_norm)
        results = retrieve_dedup(engine, slider, tid, alpha)
        if len(results) >= SCORE_TOP:
            ret = ret_align_of(engine, align, results)
            if best is None or ret > best[0]:
                best = (ret, tid, alpha)
        if tried >= N_SEED_CANDIDATES:
            break
    if best is None:  # fallback: the single lowest-alignment track with usable metadata
        for si in order.tolist():
            tid = engine.track_ids[si]
            if good_meta(engine, tid):
                return tid, peak_alpha(engine, slider, tid, te_norm)
    return best[1], best[2]


def dedup_frozen(results, seed_artist):
    sa = seed_artist.strip().lower()
    out, seen = [], set()
    for r in results:
        a = r["artist"].strip().lower()
        if a == sa or a in seen:
            continue
        seen.add(a)
        out.append(r)
    return out


def load_existing():
    if not os.path.exists(OUT):
        return {}
    src = open(OUT).read()
    try:
        return {e["id"]: e for e in json.loads(src[src.index("["): src.rindex("]") + 1])}
    except Exception:
        return {}


def main():
    engine = get_engine()
    print(f"engine ready on {engine.device}; corpus {len(engine.track_ids)} tracks")
    prev = load_existing()

    scored = []
    for cid, concept, accent in POOL:
        # regen-only mode freezes everything not listed; otherwise freeze KEEP_IDS.
        freeze = cid in prev and (cid not in REGEN_ONLY if REGEN_ONLY else cid in KEEP_IDS)
        if freeze:
            ex = dict(prev[cid])
            ex["accent"] = accent
            ex["results"] = dedup_frozen(ex["results"], ex["seed"]["artist"])
            ex.setdefault("ret_align", 0.0)
            print(f"  [keep] {concept:24s} α={ex.get('alpha')}  seed={ex['seed']['artist'][:20]:20s}  "
                  f"({len(ex['results'])} results, frozen)")
            scored.append(ex)
            continue

        te_norm, align = te_and_align(engine, concept)
        seed_id, alpha = choose_seed(engine, concept, align, te_norm)
        ex, seed_align = build_example(engine, cid, concept, accent, seed_id, alpha, align, te_norm)
        print(f"  [new ] {concept:24s} α={alpha}  seed={ex['seed']['artist'][:20]:20s}  "
              f"seed_align={seed_align:+.2f} -> ret_align={ex['ret_align']:+.2f}  ({len(ex['results'])} results)")
        scored.append(ex)

    scored.sort(key=lambda e: e["ret_align"], reverse=True)
    print("ranked:", ", ".join(f"{e['concept']}({e['ret_align']:+.2f})" for e in scored))

    header = (
        "// Real steerable-retrieval examples over music4all (MuQ-MuLan embeddings).\n"
        f"// Generated by demo/gen_examples.py from {MODEL_TAG}. ALL candidates emitted,\n"
        "// sorted by retrieved concept alignment — trim this list by hand.\n"
        "// Regenerated concepts: low-alignment seed, alpha at the steered-query peak\n"
        "// (guards against over-steering reversal), results de-duped. KEEP_IDS frozen.\n"
        "export const PLACEHOLDER = false\n\n"
        "export const EXAMPLES = "
    )
    with open(OUT, "w") as fh:
        fh.write(header)
        fh.write(json.dumps(scored, indent=2))
        fh.write("\n")
    print(f"wrote {len(scored)} examples -> {OUT}")


if __name__ == "__main__":
    main()
