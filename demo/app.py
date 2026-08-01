"""Sparse Steerable Retrieval — live demo (Hugging Face ZeroGPU Space).

Search for a seed track or draw one at random, make one or more concept sliders,
and retrieve live as the sliders move. No audio is hosted — playback is via
Spotify embeds.

ZeroGPU: the corpus + MuQ weights are fetched at startup (CPU); the text tower is
used on GPU only while a new slider is made. Live slider edits reuse cached CPU
masks.
"""
import html
import os
import random
import re

import gradio as gr
import numpy as np

# `spaces` exists only on HF ZeroGPU; shim to a no-op decorator elsewhere.
try:
    import spaces
except Exception:  # pragma: no cover
    class _Spaces:
        def GPU(self, *a, **k):
            def deco(fn):
                return fn
            return deco if not (a and callable(a[0])) else a[0]
    spaces = _Spaces()

from core import DemoEngine, load_corpus

K = 8
MAX_SLIDERS = 5
EXAMPLE_CONCEPTS = [
    "piano",
    "synthwave",
    "less vocals",
    "more drums",
    "distorted guitar",
    "dreamy and ethereal",
    "warm and intimate",
]

# --- startup (CPU): pre-cache weights so the first GPU call stays within budget -------- #
for _repo in ("OpenMuQ/MuQ-MuLan-large", "OpenMuQ/MuQ-large-msd-iter"):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(_repo)
    except Exception:
        pass

CORPUS = load_corpus()
EMB_NORM = CORPUS[0] / (np.linalg.norm(CORPUS[0], axis=1, keepdims=True) + 1e-8)
print(f"corpus loaded: {len(CORPUS[1])} tracks")

_ENGINE = None


def ensure_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DemoEngine(corpus=CORPUS, dev="cuda")
    return _ENGINE


# --- data helpers --------------------------------------------------------------------- #
def _safe(x):
    return html.escape(str(x or ""))


def _spotify(sid, height=80):
    if not sid:
        return '<div class="ssr-embed ssr-empty">Spotify preview unavailable</div>'
    sid = _safe(sid)
    return (
        f'<iframe class="ssr-embed" src="https://open.spotify.com/embed/track/{sid}?theme=0" '
        f'width="100%" height="{height}" frameBorder="0" loading="lazy" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>'
    )


def _meta(track_id):
    m = CORPUS[2].get(track_id, {})
    return m.get("title", ""), m.get("artist", ""), m.get("genre", ""), m.get("spotify", "")


def _label_for(track_id):
    title, artist, genre, _ = _meta(track_id)
    bits = [b for b in (title, artist, genre) if b]
    return f"{' — '.join(bits[:2])}{f' · {genre}' if genre and len(bits) >= 2 else ''} [{track_id}]"


def _track_from_label(label):
    if not label:
        return None
    match = re.search(r"\[([^\]]+)\]\s*$", str(label))
    return match.group(1) if match else str(label)


def _search_text(track_id):
    title, artist, genre, _ = _meta(track_id)
    return f"{title} {artist} {genre}".lower()


SEARCH_INDEX = [(tid, _search_text(tid)) for tid in CORPUS[1]]


def search_tracks(query):
    query = (query or "").strip().lower()
    if not query:
        choices = [_label_for(tid) for tid in random.sample(CORPUS[1], min(12, len(CORPUS[1])))]
    else:
        tokens = query.split()
        ranked = []
        for tid, text in SEARCH_INDEX:
            if all(tok in text for tok in tokens):
                title, artist, genre, _ = _meta(tid)
                starts = int(title.lower().startswith(query)) + int(artist.lower().startswith(query))
                ranked.append((starts, tid))
        ranked.sort(reverse=True)
        choices = [_label_for(tid) for _, tid in ranked[:12]]
    if not choices:
        return gr.update(choices=[], value=None), '<span class="ssr-muted">No matching tracks. Try artist, title, or genre.</span>'
    return gr.update(choices=choices, value=choices[0]), '<span class="ssr-muted">Select a seed from the matches.</span>'


def seed_card(track_id):
    title, artist, genre, sid = _meta(track_id)
    return (
        '<div class="ssr-seed">'
        '<div class="ssr-kicker">Seed track</div>'
        f'<div class="ssr-title">{_safe(title)}</div>'
        f'<div class="ssr-sub">{_safe(artist)} · {_safe(genre)}</div>'
        f'{_spotify(sid, 152)}'
        '</div>'
    )


def _baseline_results(seed_track_id):
    seed_idx = CORPUS[1].index(seed_track_id)
    sims = EMB_NORM @ EMB_NORM[seed_idx]
    sims[seed_idx] = -1e9
    idx = np.argsort(-sims)[:K]
    out = []
    for i in idx:
        title, artist, genre, sid = _meta(CORPUS[1][int(i)])
        out.append({
            "title": title,
            "artist": artist,
            "genre": genre,
            "spotify": sid,
            "affinity": float(sims[int(i)]),
        })
    return out


def results_html(results):
    cards = []
    for i, r in enumerate(results, 1):
        cards.append(
            '<div class="ssr-card">'
            '<div class="ssr-row">'
            f'<span class="ssr-rank">{i}</span>'
            '<span class="ssr-track">'
            f'<span class="ssr-title">{_safe(r.get("title"))}</span>'
            f'<span class="ssr-sub">{_safe(r.get("artist"))} · {_safe(r.get("genre"))}</span>'
            '</span>'
            f'<span class="ssr-aff">{float(r.get("affinity", 0.0)):.2f}</span>'
            '</div>'
            f'{_spotify(r.get("spotify"))}'
            '</div>'
        )
    return '<div class="ssr-results">' + "".join(cards) + "</div>"


def _active_pairs(concepts, alphas):
    concepts = concepts or []
    return [
        (str(concept), float(alpha or 0.0))
        for concept, alpha in zip(concepts, alphas)
        if str(concept).strip()
    ]


def _status_html(concepts, alphas, *, prefix=None):
    concepts = concepts or []
    chips = []
    for concept, alpha in _active_pairs(concepts, alphas):
        tone = "pos" if alpha > 0 else "neg" if alpha < 0 else "zero"
        chips.append(f'<span class="ssr-chip {tone}">{_safe(concept)} <b>{alpha:+.1f}</b></span>')
    head = f'<span>{_safe(prefix)}</span>' if prefix else '<span>Live retrieval query</span>'
    return '<div class="ssr-note">' + head + '<div class="ssr-chiprow">' + "".join(chips) + '</div></div>'


def _slider_components(concepts, values=None):
    values = list(values or [])
    updates = []
    for i in range(MAX_SLIDERS):
        if i < len(concepts):
            concept = concepts[i]
            value = float(values[i]) if i < len(values) else 0.0
            label = (
                '<div class="ssr-slider-label">'
                f'<span>{_safe(concept)}</span>'
                f'<span class="ssr-alpha">α {value:+.1f}</span>'
                '</div>'
            )
            updates.extend([
                gr.update(value=label, visible=True),
                gr.update(value=value, visible=True, label=concept),
            ])
        else:
            updates.extend([
                gr.update(value="", visible=False),
                gr.update(value=0.0, visible=False, label=f"Concept {i + 1}"),
            ])
    return updates


def random_seed():
    tid = random.choice(CORPUS[1])
    return tid, seed_card(tid), _status_html([], [], prefix="Showing nearest neighbours for the seed."), results_html(_baseline_results(tid))


def select_seed(choice):
    tid = _track_from_label(choice)
    if not tid:
        return None, "", '<span class="ssr-muted">Search for a track or choose random.</span>', ""
    return tid, seed_card(tid), _status_html([], [], prefix="Showing nearest neighbours for the seed."), results_html(_baseline_results(tid))


@spaces.GPU(duration=120)
def make_slider(seed_track_id, concept, concepts, *alphas):
    if not seed_track_id:
        return (
            concepts or [],
            '<span class="ssr-muted">Pick a seed track first.</span>',
            "",
            *_slider_components(concepts or [], alphas),
        )
    concept = (concept or "").strip()
    concepts = list(concepts or [])
    values = [float(alphas[i] or 0.0) for i in range(min(len(concepts), len(alphas)))]
    if not concept:
        return concepts, '<span class="ssr-muted">Type a concept, then make a slider.</span>', results_html(_baseline_results(seed_track_id)), *_slider_components(concepts, alphas)
    if concept.lower() not in {c.lower() for c in concepts}:
        if len(concepts) >= MAX_SLIDERS:
            msg = f"Maximum of {MAX_SLIDERS} sliders reached."
            return concepts, f'<span class="ssr-muted">{msg}</span>', results_html(_baseline_results(seed_track_id)), *_slider_components(concepts, alphas)
        concepts.append(concept)
        values.append(0.0)
    while len(values) < len(concepts):
        values.append(0.0)

    eng = ensure_engine()
    support = len(eng._slider(concept))
    results = eng.multi_steer_and_retrieve(seed_track_id, list(zip(concepts, values)), k=K)
    note = _status_html(concepts, values, prefix=f"Slider ready: {concept} uses {support} sparse features.")
    return concepts, note, results_html(results), *_slider_components(concepts, values)


def live_retrieve(seed_track_id, concepts, *alphas):
    concepts = list(concepts or [])
    if not seed_track_id:
        return '<span class="ssr-muted">Pick a seed track first.</span>', "", *_slider_components(concepts, alphas)
    if not concepts:
        return (
            _status_html([], [], prefix="Showing nearest neighbours for the seed."),
            results_html(_baseline_results(seed_track_id)),
            *_slider_components(concepts, alphas),
        )
    pairs = _active_pairs(concepts, alphas)
    eng = ensure_engine()
    results = eng.multi_steer_and_retrieve(seed_track_id, pairs, k=K)
    return _status_html(concepts, alphas), results_html(results), *_slider_components(concepts, alphas)


CSS = """
:root {
  --ssr-purple: #7b3ff2;
  --ssr-green: #27995f;
  --ssr-orange: #e88a2a;
  --ssr-pink: #df6b96;
  --ssr-ink: #171717;
  --ssr-muted: #737373;
  --ssr-line: rgba(23, 23, 23, 0.12);
  --ssr-glass: rgba(255, 255, 255, 0.72);
}
body, .gradio-container {
  background: radial-gradient(70% 45% at 50% 0%, rgba(123, 63, 242, 0.11), transparent 72%), #ffffff !important;
  color: var(--ssr-ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding-left: clamp(16px, 3vw, 44px) !important;
  padding-right: clamp(16px, 3vw, 44px) !important;
}
footer, .api-docs { display: none !important; }
.ssr-hero {
  padding: 20px 4px 16px;
}
.ssr-hero h1 {
  margin: 0;
  font-size: clamp(34px, 5vw, 62px);
  letter-spacing: 0;
  line-height: 1.02;
  font-weight: 760;
}
.ssr-hero p {
  max-width: 760px;
  margin: 16px 0 0;
  color: #555;
  font-size: 17px;
  line-height: 1.62;
}
.ssr-kicker {
  color: var(--ssr-purple);
  font-size: 12px;
  font-weight: 760;
  letter-spacing: .14em;
  text-transform: uppercase;
}
.ssr-panel, .ssr-seed, .ssr-card {
  border: 1px solid var(--ssr-line);
  background: var(--ssr-glass);
  box-shadow: 0 18px 55px rgba(23, 23, 23, 0.07);
  backdrop-filter: blur(18px);
}
.ssr-panel {
  border-radius: 22px !important;
  padding: 18px !important;
  gap: 14px !important;
}
.ssr-panel,
.ssr-panel > *,
.ssr-panel .form,
.ssr-panel .block,
.ssr-panel .wrap,
.ssr-panel .gradio-row,
.ssr-panel .gradio-column {
  background-color: transparent !important;
}
.ssr-panel .form,
.ssr-panel .block {
  border: 0 !important;
  box-shadow: none !important;
}
.ssr-panel h3 {
  margin: 0 0 2px !important;
  font-size: 15px !important;
  font-weight: 760 !important;
  letter-spacing: 0 !important;
}
.ssr-panel label span {
  color: var(--ssr-purple) !important;
  font-size: 12px !important;
  font-weight: 680 !important;
}
.ssr-seed {
  padding: 16px;
  border-radius: 20px;
}
.ssr-title {
  display: block;
  color: var(--ssr-ink);
  font-weight: 720;
  line-height: 1.15;
}
.ssr-seed .ssr-title { margin-top: 8px; font-size: 22px; }
.ssr-sub {
  display: block;
  margin-top: 3px;
  color: var(--ssr-muted);
  font-size: 13px;
}
.ssr-embed {
  margin-top: 12px;
  border: 0;
  border-radius: 14px;
  background: #f5f5f5;
}
.ssr-empty {
  min-height: 76px;
  display: grid;
  place-items: center;
  color: #9b9b9b;
  font-size: 12px;
}
.ssr-results {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.ssr-card {
  padding: 12px;
  border-radius: 18px;
}
.ssr-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.ssr-track { min-width: 0; flex: 1; }
.ssr-card .ssr-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
}
.ssr-rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  flex: none;
  border-radius: 999px;
  background: var(--ssr-purple);
  color: #fff;
  font-size: 11px;
  font-weight: 760;
}
.ssr-aff {
  color: #a3a3a3;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.ssr-note {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  color: #525252;
  font-size: 13px;
}
.ssr-chiprow {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.ssr-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 9px;
  border-radius: 999px;
  border: 1px solid rgba(123, 63, 242, 0.18);
  background: rgba(123, 63, 242, 0.07);
  color: var(--ssr-purple);
}
.ssr-chip.pos { color: var(--ssr-green); border-color: rgba(39, 153, 95, .22); background: rgba(39, 153, 95, .08); }
.ssr-chip.neg { color: var(--ssr-pink); border-color: rgba(223, 107, 150, .22); background: rgba(223, 107, 150, .08); }
.ssr-chip.zero { color: var(--ssr-purple); }
.ssr-muted { color: var(--ssr-muted); font-size: 13px; }
.ssr-slider-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin: 3px 0 -4px;
  color: var(--ssr-purple);
  font-weight: 720;
}
.ssr-alpha {
  color: #a78bfa;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.gradio-container .form,
.gradio-container .block {
  border-color: rgba(23, 23, 23, 0.08) !important;
  border-radius: 16px !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  border-radius: 12px !important;
  border-color: rgba(23, 23, 23, 0.10) !important;
  box-shadow: none !important;
}
.gradio-container button {
  border-radius: 999px !important;
  font-weight: 650 !important;
  min-height: 42px !important;
  height: 42px !important;
  box-shadow: 0 10px 24px rgba(23, 23, 23, 0.08) !important;
}
.ssr-panel button {
  align-self: end !important;
  padding-left: 18px !important;
  padding-right: 18px !important;
}
.gradio-container button.primary {
  background: var(--ssr-ink) !important;
  border-color: var(--ssr-ink) !important;
}
input[type='range'] {
  accent-color: var(--ssr-purple);
}
"""

THEME = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="neutral",
    neutral_hue="neutral",
).set(
    body_background_fill="#ffffff",
    button_primary_background_fill="#171717",
    button_primary_background_fill_hover="#000000",
    button_primary_text_color="#ffffff",
    block_radius="18px",
    input_radius="12px",
)

with gr.Blocks(title="Sparse Steerable Retrieval", css=CSS, theme=THEME) as demo:
    gr.HTML(
        '<div class="ssr-hero">'
        '<div class="ssr-kicker">Sparse Steerable Retrieval</div>'
        '<h1>Steer music retrieval with concept sliders.</h1>'
        '<p>Search for a seed track or draw one at random. Make sliders from free-form concepts, '
        'then move several at once: retrieval updates live from the combined sparse edit.</p>'
        '</div>'
    )

    seed_state = gr.State()
    slider_state = gr.State([])

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Column(elem_classes=["ssr-panel"]):
                gr.Markdown("### 1. Choose a seed")
                with gr.Row():
                    search = gr.Textbox(
                        label="Search music4all",
                        placeholder="artist, title, or genre",
                        scale=4,
                    )
                    search_btn = gr.Button("Search", variant="secondary", scale=1)
                    random_btn = gr.Button("Random seed", variant="secondary", scale=1)
                matches = gr.Dropdown(label="Search results", choices=[], interactive=True)
                pick_btn = gr.Button("Use selected seed", variant="primary")
                search_note = gr.HTML('<span class="ssr-muted">Search for a track, or start from a random seed.</span>')
                seed_html = gr.HTML()

            with gr.Column(elem_classes=["ssr-panel"]):
                gr.Markdown("### 2. Make concept sliders")
                with gr.Row():
                    concept = gr.Textbox(
                        label="Concept",
                        placeholder="piano, synthwave, less vocals, more drums",
                        scale=4,
                    )
                    make_btn = gr.Button("Make slider", variant="primary", scale=1)
                gr.Examples(EXAMPLE_CONCEPTS, inputs=concept, label="Try concepts")

                slider_labels = []
                slider_controls = []
                for i in range(MAX_SLIDERS):
                    label = gr.HTML(visible=False)
                    control = gr.Slider(
                        minimum=-3.0,
                        maximum=3.0,
                        value=0.0,
                        step=0.1,
                        label=f"Concept {i + 1}",
                        visible=False,
                        interactive=True,
                    )
                    slider_labels.append(label)
                    slider_controls.append(control)

                note = gr.HTML()

        with gr.Column(scale=5):
            with gr.Column(elem_classes=["ssr-panel"]):
                gr.Markdown("### 3. Live retrieval")
                results_out = gr.HTML()

    search_btn.click(search_tracks, inputs=[search], outputs=[matches, search_note])
    search.submit(search_tracks, inputs=[search], outputs=[matches, search_note])
    random_btn.click(random_seed, outputs=[seed_state, seed_html, note, results_out])
    pick_btn.click(select_seed, inputs=[matches], outputs=[seed_state, seed_html, note, results_out])
    matches.change(select_seed, inputs=[matches], outputs=[seed_state, seed_html, note, results_out])

    slider_outputs = [slider_state, note, results_out]
    for label, control in zip(slider_labels, slider_controls):
        slider_outputs.extend([label, control])

    make_btn.click(
        make_slider,
        inputs=[seed_state, concept, slider_state, *slider_controls],
        outputs=slider_outputs,
    )

    live_outputs = [note, results_out]
    for label, control in zip(slider_labels, slider_controls):
        live_outputs.extend([label, control])
    for control in slider_controls:
        control.change(
            live_retrieve,
            inputs=[seed_state, slider_state, *slider_controls],
            outputs=live_outputs,
        )

    demo.load(random_seed, outputs=[seed_state, seed_html, note, results_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
