from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _normalize_for_hash(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return {str(key): _normalize_for_hash(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_for_hash(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if torch is not None and isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        _normalize_for_hash(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def build_cache_path(
    cache_root: str | Path,
    eval_config: Any,
    run_id: str,
    model_config: Any,
    suffix: str = "edit.pkl",
) -> tuple[str, str, Path]:
    eval_hash = stable_hash(eval_config)
    model_hash = stable_hash({"run_id": run_id, "model_config": model_config})
    cache_path = Path(cache_root) / f"{eval_hash}_{model_hash}_{suffix}"
    return eval_hash, model_hash, cache_path


def load_cached_frame(cache_path: str | Path) -> pd.DataFrame:
    return pd.read_pickle(Path(cache_path))


def save_cached_frame(df: pd.DataFrame, cache_path: str | Path) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    return cache_path


def build_cache_metadata_path(cache_path: str | Path, metadata_filename: str = "hash.json") -> Path:
    cache_path = Path(cache_path)
    cache_stem = cache_path.name
    if cache_path.suffix:
        cache_stem = cache_path.name[: -len(cache_path.suffix)]
    return cache_path.parent / f"{cache_stem}.{metadata_filename}"


def save_cache_metadata(metadata: Any, cache_path: str | Path, metadata_filename: str = "hash.json") -> Path:
    metadata_path = build_cache_metadata_path(cache_path, metadata_filename=metadata_filename)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(_normalize_for_hash(metadata), f, indent=2, sort_keys=True)
    return metadata_path
