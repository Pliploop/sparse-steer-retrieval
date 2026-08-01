"""Stage and push the live demo to Hugging Face: model repo, corpus dataset, and Space.

Run AFTER `hf auth login` (a write token). Idempotent-ish: repos are created with
exist_ok and folders re-uploaded.

    python demo/hf/push_all.py                 # push everything
    python demo/hf/push_all.py --space-only    # just re-push the Space (fast iteration)

Repos (edit USER / *_REPO below to taste):
  model   Pliploop/steerable-retrieval-sae      L0-5 .. L0-100 (last.ckpt + config.yaml)
  dataset Pliploop/steerable-retrieval-corpus   corpus.npz + meta.json (PRIVATE)
  space   Pliploop/steerable-retrieval          the ZeroGPU Gradio app
"""
import argparse
import os
import shutil

from huggingface_hub import HfApi, get_token

USER = "Pliploop"
MODEL_REPO = f"{USER}/steerable-retrieval-sae"
CORPUS_REPO = f"{USER}/steerable-retrieval-corpus"
SPACE_REPO = f"{USER}/steerable-retrieval"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEMO = os.path.join(REPO, "demo")
OUT = os.path.join(DEMO, "hf", "_staged")

# L0 -> Lightning run dir under logs/xps
RUNS = {5: "2d0e39fb", 10: "590f2aba", 20: "66a0caf9", 50: "cec15d5c", 100: "1e42beb7"}
DEFAULT_L0 = 20


def stage_model():
    root = os.path.join(OUT, "model")
    shutil.rmtree(root, ignore_errors=True)
    for l0, run in RUNS.items():
        d = os.path.join(root, f"L0-{l0}")
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(REPO, "logs/xps", run, "checkpoints/last.ckpt"), os.path.join(d, "last.ckpt"))
        shutil.copy(os.path.join(REPO, "logs/xps", run, ".hydra/config.yaml"), os.path.join(d, "config.yaml"))
    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write(
            "---\nlicense: mit\ntags: [music, retrieval, sparse-autoencoder, muq]\n---\n\n"
            "# Sparse Steerable Retrieval — BatchTopK SAEs\n\n"
            "BatchTopK SAEs trained on MuQ-MuLan music4all audio embeddings (512-d), one per "
            "sparsity level L0. Each `L0-*/` holds `last.ckpt` (encoder/decoder weights) + "
            "`config.yaml`. Load via `steerable_retrieval.steer.load_steerable_sae(repo, subfolder='L0-20')`.\n"
        )
    return root


def stage_space():
    root = os.path.join(OUT, "space")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    for f in ("app.py", "core.py", "requirements.txt", "README.md"):
        shutil.copy(os.path.join(DEMO, f), os.path.join(root, f))
    # vendor the package so the Space is self-contained (no pip-from-git needed)
    shutil.copytree(
        os.path.join(REPO, "steerable_retrieval"),
        os.path.join(root, "steerable_retrieval"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-only", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    token = get_token()
    if not token:
        raise SystemExit("No HF token found. Run `hf auth login` first.")

    if not args.space_only:
        print(f"== model repo {MODEL_REPO} ==")
        api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=stage_model())

        print(f"== corpus dataset {CORPUS_REPO} (private) ==")
        api.create_repo(CORPUS_REPO, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=CORPUS_REPO, repo_type="dataset",
            folder_path=os.path.join(DEMO, "corpus"),
            allow_patterns=["corpus.npz", "meta.json"],
        )

    print(f"== space {SPACE_REPO} (gradio / ZeroGPU) ==")
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=SPACE_REPO, repo_type="space", folder_path=stage_space())

    # Space config: variables + the token secret (to read the private corpus).
    for k, v in {
        "SSR_CHECKPOINT": MODEL_REPO,
        "SSR_L0": f"L0-{DEFAULT_L0}",
        "SSR_CORPUS_REPO": CORPUS_REPO,
        "SSR_DEVICE": "cuda",
    }.items():
        api.add_space_variable(SPACE_REPO, k, v)
    api.add_space_secret(SPACE_REPO, "HF_TOKEN", token)

    try:
        api.request_space_hardware(SPACE_REPO, "zero-a10g")  # ZeroGPU
        print("requested ZeroGPU hardware")
    except Exception as e:
        print(f"!! set hardware to ZeroGPU manually in Space settings ({e})")

    print(f"\nDone. Space: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
