---
title: Sparse Steerable Retrieval
emoji: 🎚️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Build live concept sliders for dense music retrieval
---

# Sparse Steerable Retrieval — live demo

Open-vocabulary concept control for dense music retrieval. Search for a seed track
or draw one at random, make one or more free-form concept sliders, then move the
sliders together. The query embedding is edited through sparse inversion in a
trained BatchTopK SAE (MuQ-MuLan / music4all), and nearest neighbours are retrieved
live after each slider move. No audio is hosted — playback is via Spotify embeds.

Runs on **ZeroGPU**: the corpus + MuQ weights are fetched at startup; the text tower is
used on GPU only while a new slider is instantiated. Sparse inversion, slider edits,
and retrieval run on CPU from cached slider masks.

## Configuration (Space variables / secrets)

- `SSR_CHECKPOINT` — HF model repo id, e.g. `Pliploop/steerable-retrieval-sae` (variable)
- `SSR_L0` — subfolder to load, e.g. `L0-20` (variable)
- `SSR_CORPUS_REPO` — HF **dataset** repo id with `corpus.npz` + `meta.json` (variable)
- `SSR_DEVICE` — `cuda` (variable)
- `HF_TOKEN` — a read token (secret) so the Space can read the private corpus dataset

## Layout

- `app.py` — the Gradio ZeroGPU app: seed search, slider creation, live retrieval.
- `core.py` — engine: load model + corpus, combine sparse slider edits, retrieve.
- `steerable_retrieval/` — vendored package (steering API + SAE + MuQ encoder).

This Space is published by `hf/push_all.py` in the project repo.
