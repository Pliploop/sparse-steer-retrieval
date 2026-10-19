# Music Lenses

A standalone static page at `/sparse-steer-retrieval/lenses/`, linked from the
project navigation and hero. `npm run build` copies it into `dist/lenses/`.
The existing GitHub Pages workflow publishes it with the rest of the site.
There is no app server, API key or audio file in this package.

## What runs in the browser

All **109,269 Music4All tracks** can be starting songs. The eight built-in lenses
include controls and retain 80% of member variance. Each has two basis variants:
original MuQ-MuLan text vectors, and decoded vectors from the original Adam
sparse inversion (L0-20 run 66a0caf9, 500 steps, packaged prior).

The selected view downloads its normalized float32 projected embeddings and a
browser worker performs cosine search over the entire corpus. Matrices are
loaded lazily and bounded by a three-view cache. Full-space top ten neighbours
and scores are precomputed for **every** song, avoiding a 224 MB raw-corpus
transfer. These are genuine corpus results, not curated or fabricated examples.
The suggested artists are conveniences; the search field, shuffle and the explicit star action in a map card can select any track.

The total data package is about 184 MB, divided across independent files. A
visitor downloads the catalog and the selected view, then caption and affinity
shards as needed, rather than the whole package. There is no quantization of
retrieval vectors or prompt affinities. Floating-point score differences of a
few millionths between JavaScript and BLAS are expected.

The flat t-SNE map is fitted to 5,000 fixed tracks per view. Other tracks are
interpolated from eight neighbours. Only the fixed sample and the current
query/results are drawn. Coordinates are approximate display positions, not
retrieval distances. Optional soft-density group labels reflect caption/metadata profiles; they are separate from retrieval distances and may overlap.

Track captions and original MuQ-MuLan prompt affinities are in 512-track shards.
Attributes use the largest minimum affinity shared by query and candidate,
matching the research app. They remain model estimates. Neither custom axes nor
variance/rank editing are exposed here; those stay in the local research app.

## Playback

The top-centre glass transport contains play/pause, activity bars, title and
artist. Activity bars and a single compositor-animated dot pulse follow the
actual playback state and current map colour. Bars indicate activity, not
analysed Spotify audio. Cards open on hover/focus/tap; only their explicit star
button changes the starting song.

`playback.json` defaults to `audio: null`. Buttons then offer an external Spotify
link or an explicitly requested **visible**, unmodified 152-pixel Spotify embed.
No widget is loaded by default and none is hidden during playback. Spotify
controls availability and autoplay; its preview may cover a different passage
from the Music4All clip. No Spotify credentials or preview scraping are used.

The same assets support a separate private exact-clip host. Its dynamic
`playback.json` supplies `audio.base_url`, `audio.extension` and `audio.duration`.
The browser plays those original 30-second excerpts directly, without Spotify.
The source checkout's `concept_lab/hf/` contains the private Docker Space and
read-only private dataset-volume deployment. Private audio is not included here.

- https://developer.spotify.com/documentation/embeds/references/iframe-api
- https://developer.spotify.com/documentation/embeds/terms

## Regeneration

The source research checkout is the sibling `sparse-steerable-retrieval` repo.
From that checkout, on a CPU allocation with the documented dataset and model
caches:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /data/home/acw749/conda-envs/steer/bin/python -m concept_lab.export_static \
  --output ../sparse-steer-retrieval/website/public/lenses
```

`concept_lab/export_static.sbatch` performs the data export on computeshort.
The all-song full-space graph takes about 5.5 minutes on four CPUs on this
machine; its intermediate cache is resumable. `--assets-only` refreshes the JS,
HTML and fonts without recomputing results. The static build itself needs only
this checkout and the checked-in package, not Python or either model.

## Format and provenance

`data/manifest.json` records the model identity, source-corpus count, settings,
version and paths. `catalog.json` contains track IDs/titles/artist indices and
Spotify IDs. Per-view JSON files record eigenvalues, prompts and inversion
diagnostics. Binary matrices are row-major little-endian float32; memberships
are uint8 and full-space neighbour indices are uint32. Rows follow the catalog.
`data/reference.json` contains independent Python ranking/score reference cases.

To run browser QA, serve the **built** site under its configured base path, then
from the source research checkout:

```sh
LENS_TEST_URL=http://127.0.0.1:5190/sparse-steer-retrieval/lenses/ \
  node concept_lab/tests/packaged.mjs
```

The test checks worker rankings against Python, arbitrary song selection,
metadata/attribute hydration, both bases, full-space comparison, bounded cards,
visible Spotify integration and mobile layout. The project page also retains
its `npm run ssr-check` validation.
