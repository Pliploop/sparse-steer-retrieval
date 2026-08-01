from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from steerable_retrieval.callbacks.energy import load_state_dict_any
from steerable_retrieval.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _silence_huggingface_loading() -> None:
    """Reduce HuggingFace/Transformers load-time noise (progress bars + logs)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_config(cfg: DictConfig, output_dir: str, filename: str = "config.yaml") -> str:
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    OmegaConf.save(cfg, out_path, resolve=True)
    return out_path


def resolve_checkpoint_path(model_source_cfg: DictConfig) -> str:
    ckpt = model_source_cfg.get("checkpoint_source")
    if ckpt is None:
        # Backward compatibility with older experiment configs.
        ckpt = model_source_cfg.get("checkpoint_path")
    if ckpt:
        return str(ckpt)

    run_id = model_source_cfg.get("run_id")
    if run_id is None:
        run_id = model_source_cfg.get("config_source")
    if run_id is None:
        run_id = model_source_cfg.get("wandb_run_id")
    if not run_id:
        raise ValueError(
            "Missing model_source.run_id (or compatible fallback). "
            "Set model_source.checkpoint_source explicitly or provide a run id for auto checkpoint resolution."
        )
    run_id = str(run_id)
    if ".yaml" in run_id:
        raise ValueError(
            "Auto checkpoint resolution expects model_source.config_source to be a run id, "
            "not a YAML path. Set model_source.checkpoint_source explicitly."
        )

    s3_prefix = str(model_source_cfg.get("checkpoint_s3_prefix", "s3://your-bucket/checkpoints")).rstrip("/")
    parsed = urlparse(s3_prefix)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(
            f"Invalid model_source.checkpoint_s3_prefix '{s3_prefix}'. Expected an s3:// URI."
        )
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    key_prefix = f"{prefix}/{run_id}/checkpoints/".lstrip("/")

    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required to auto-resolve checkpoint path from S3.") from exc

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    candidates = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key", ""))
            filename = key.rsplit("/", 1)[-1]
            if "epoch" in filename and (key.endswith(".pt") or key.endswith(".ckpt")):
                candidates.append(obj)

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint file found under "
            f"s3://{bucket}/{key_prefix} matching '*epoch*.pt' or '*epoch*.ckpt'."
        )

    latest = max(candidates, key=lambda x: x.get("LastModified"))
    latest_key = str(latest["Key"])
    resolved_ckpt = f"s3://{bucket}/{latest_key}"
    log.info(f"Auto-resolved checkpoint from run id '{run_id}': {resolved_ckpt}")
    return resolved_ckpt


def resolve_run_id(run_id: str, entity: Optional[str], project: Optional[str]) -> str:
    if not entity or not project:
        raise ValueError(
            "wandb.entity and wandb.project are required to resolve run id via "
            "config.sagemaker_job_name."
        )
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("wandb is required to resolve model_source.config_source.") from exc

    api = wandb.Api()
    runs = api.runs(
        path=f"{entity}/{project}",
        filters={"config.sagemaker_job_name": run_id},
        per_page=1,
    )
    runs = list(runs)
    if not runs:
        raise ValueError(
            f"No W&B run found in {entity}/{project} with config.sagemaker_job_name='{run_id}'."
        )
    resolved = str(runs[0].id)
    log.info(
        f"Resolved model_source.config_source '{run_id}' to W&B run id '{resolved}' "
        f"via config.sagemaker_job_name."
    )
    return resolved


def _fetch_model_cfg_from_wandb(run_id: str, entity: Optional[str], project: Optional[str]) -> DictConfig:
    resolved_run_id = resolve_run_id(run_id=run_id, entity=entity, project=project)
    run_path = f"{entity}/{project}/{resolved_run_id}"
    import wandb
    api = wandb.Api()
    run = api.run(run_path)
    run_cfg = OmegaConf.create(run.config)
    model_cfg = run_cfg.get("model")
    if model_cfg is None:
        raise KeyError(f"No 'model' key found in wandb run config for {run_path}.")
    model_cfg = OmegaConf.create(model_cfg)
    try:
        OmegaConf.resolve(model_cfg)
    except Exception:
        log.warning(
            "Could not fully resolve interpolations inside wandb run model config. "
            "Proceeding with unresolved values."
        )
    return OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=False))


def _fetch_yaml_cfg_from_s3(s3_uri: str) -> DictConfig:
    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required to read YAML model config from S3.") from exc

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI for model_source.config_source: {s3_uri}")

    client = boto3.client("s3")
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    cfg = OmegaConf.create(body)
    OmegaConf.resolve(cfg)
    return cfg


def _fetch_model_cfg_from_yaml(config_source: str) -> DictConfig:
    if config_source.startswith("s3://"):
        cfg = _fetch_yaml_cfg_from_s3(config_source)
    else:
        cfg = OmegaConf.load(config_source)
        OmegaConf.resolve(cfg)

    model_cfg = cfg.get("model")
    # Accept either a full experiment config (with "model") or a model-only config.
    if model_cfg is None:
        model_cfg = cfg
    return OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))


def resolve_model_from_source(cfg: DictConfig) -> str:
    model_source_cfg = cfg.get("model_source")
    if model_source_cfg is None:
        raise ValueError("Missing cfg.model_source.")

    run_id = model_source_cfg.get("run_id")
    config_source = model_source_cfg.get("config_source")
    if config_source is None:
        # Backward compatibility with older experiment configs.
        config_source = model_source_cfg.get("wandb_run_id")
    if not run_id and not config_source:
        raise ValueError(
            "model_source.run_id must be set (preferred), or set model_source.config_source "
            "to a W&B id or a .yaml path (local/S3)."
        )

    if run_id:
        run_id = str(run_id)
        model_cfg = _fetch_model_cfg_from_wandb(
            run_id=run_id,
            entity=cfg.get("wandb", {}).get("entity"),
            project=cfg.get("wandb", {}).get("project"),
        )
        resolved_source = run_id
    else:
        config_source = str(config_source)
        if ".yaml" in config_source:
            model_cfg = _fetch_model_cfg_from_yaml(config_source)
        else:
            model_cfg = _fetch_model_cfg_from_wandb(
                run_id=config_source,
                entity=cfg.get("wandb", {}).get("entity"),
                project=cfg.get("wandb", {}).get("project"),
            )
        resolved_source = config_source

    # Apply optional local/runtime overrides after fetching model config from source.
    # This is useful for local CPU smoke tests when source configs are CUDA-oriented.
    post_overrides = cfg.get("model_post_overrides", {})
    if not torch.cuda.is_available():
        model_cfg.sae_encoder.device = 'cpu'
        model_cfg.sae_decoder.device = 'cpu'
        model_cfg.audio_encoder.device = 'cpu'
        model_cfg.text_encoder.device = 'cpu'
    if post_overrides:
        model_cfg = OmegaConf.merge(model_cfg, post_overrides)

    cfg.model = model_cfg
    return resolved_source


def instantiate_model_and_load(cfg: DictConfig, device: torch.device):
    _silence_huggingface_loading()
    model = hydra.utils.instantiate(cfg.model)
    ckpt_path = resolve_checkpoint_path(cfg.model_source)
    state_dict = load_state_dict_any(ckpt_path, map_location="cpu")
    model_state_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()
    matched_keys = model_state_keys.intersection(checkpoint_keys)
    if not matched_keys:
        raise RuntimeError(
            "Checkpoint appears incompatible: zero matching keys between model and checkpoint state_dict."
        )
    matched_key_names = sorted(matched_keys)
    load_result = model.load_state_dict(state_dict, strict=False)
    load_report = {
        "checkpoint_keys": len(checkpoint_keys),
        "model_keys": len(model_state_keys),
        "matched_keys": len(matched_keys),
        "matched_key_names": matched_key_names,
        "missing_keys": len(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": len(getattr(load_result, "unexpected_keys", [])),
    }
    setattr(model, "_checkpoint_load_report", load_report)
    model = model.to(device)
    model.eval()
    preview_limit = 30
    matched_preview = matched_key_names[:preview_limit]
    remaining = len(matched_key_names) - len(matched_preview)
    preview_suffix = f" ... (+{remaining} more)" if remaining > 0 else ""
    log.info(
        f"Loaded model from checkpoint: {ckpt_path} | "
        f"matched={load_report['matched_keys']}/{load_report['model_keys']} "
        f"missing={load_report['missing_keys']} unexpected={load_report['unexpected_keys']} "
        f"| matched_keys={matched_preview}{preview_suffix}"
    )
    return model, ckpt_path


def read_vocab_lines(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f if line.strip()]
    if not concepts:
        raise ValueError(f"No concepts found in vocabulary file: {vocab_path}")
    return concepts


def resolve_callable(path_str: str) -> Callable[..., Any]:
    if ":" in path_str:
        module_name, fn_name = path_str.split(":", 1)
    else:
        module_name, fn_name = path_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Resolved object is not callable: {path_str}")
    return fn


def maybe_start_wandb(cfg: DictConfig, job_type: str, name: str, output_dir: str):
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", True):
        return None
    try:
        import wandb
    except Exception:
        log.warning("wandb not available, skipping remote logging.")
        return None

    run = wandb.init(
        project=wb_cfg.get("project", "SpaMR"),
        entity=wb_cfg.get("entity"),
        name=name,
        group=wb_cfg.get("group"),
        job_type=job_type,
        tags=list(wb_cfg.get("tags", [])),
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        import wandb

        wandb.finish()
    except Exception:
        return


def save_json(path: str, payload: Dict[str, Any]) -> None:
    import json

    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


