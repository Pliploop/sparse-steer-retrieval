from __future__ import annotations

import importlib
import math
import os
import re
from functools import partial
from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F

from steerable_retrieval.callbacks.energy import BridgeScoreCallback, get_dictionary_from_lightningsae, load_state_dict_any
from steerable_retrieval.experiments.common import ensure_dir, read_vocab_lines
from steerable_retrieval.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return slug or "na"


def _flatten_params(prefix: str, value: Any, items: list[tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_params(child_prefix, value[key], items)
        return
    items.append((prefix or "param", str(value)))


def _auto_experiment_id(cfg: DictConfig) -> str:
    extractor_cfg = cfg.concepts_experiment.get("extractor")
    params_cfg = extractor_cfg.get("params", {}) if extractor_cfg is not None else {}
    params = OmegaConf.to_container(params_cfg, resolve=True) if params_cfg is not None else {}
    if not isinstance(params, dict):
        params = {"value": params}

    method = params.get("selection_method")
    if method is None:
        target = str(extractor_cfg.get("target", "extract")) if extractor_cfg is not None else "extract"
        method = target.rsplit(".", 1)[-1]
    method_slug = _slugify(method)

    flat_items: list[tuple[str, str]] = []
    _flatten_params("", params, flat_items)
    parts = [f"{_slugify(k)}-{_slugify(v)}" for k, v in flat_items if str(v) != ""]
    suffix = "__".join(parts)
    out = f"{method_slug}__{suffix}" if suffix else method_slug
    return out[:180]


def _resolve_output_dir(cfg: DictConfig, output_dir: str, source_run_id: str) -> tuple[str, str, str]:
    run_id = cfg.get("model_source", {}).get("run_id") or source_run_id or "unknown_run"
    run_slug = _slugify(run_id)
    experiment_id_cfg = cfg.get("experiment_id")
    experiment_id = _slugify(experiment_id_cfg) if experiment_id_cfg else _auto_experiment_id(cfg)
    full_output_dir = ensure_dir(os.path.join(output_dir, run_slug, experiment_id))
    return full_output_dir, run_slug, experiment_id


def _sort_unique_by_abs_desc(indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return indices.to(dtype=torch.long)
    uniq = torch.unique(indices.to(dtype=torch.long))
    abs_vals = values[uniq].abs()
    order = torch.argsort(abs_vals, descending=True)
    return uniq[order]


def _cap_indices(indices: torch.Tensor, max_selected: int | None) -> torch.Tensor:
    if max_selected is None:
        return indices
    if max_selected <= 0:
        return indices[:0]
    return indices[: int(max_selected)]


def get_selected_indices_abs_mass(
    cosine_similarities: torch.Tensor,
    *,
    eta: float = 0.95,
    max_selected: int | None = None,
):
    abs_cosine = cosine_similarities.abs()
    selected_indices = torch.empty(0, dtype=torch.long, device=cosine_similarities.device)
    k95 = 0
    if abs_cosine.numel() > 0:
        sorted_abs_values, sorted_abs_indices = torch.sort(abs_cosine, descending=True)
        total_abs_mass = sorted_abs_values.sum()
        if float(total_abs_mass.item()) > 0.0:
            cumsum_abs = torch.cumsum(sorted_abs_values, dim=0)
            threshold = eta * total_abs_mass
            cutoff_candidates = torch.nonzero(cumsum_abs >= threshold, as_tuple=False)
            cutoff = int(cutoff_candidates[0].item()) + 1 if cutoff_candidates.numel() > 0 else sorted_abs_values.numel()
        else:
            cutoff = 0
        selected_indices = sorted_abs_indices[:cutoff]
        k95 = cutoff
    selected_indices = _cap_indices(selected_indices, max_selected=max_selected)
    return selected_indices, {
        "method": "abs_mass",
        "k95": int(k95),
        "eta": float(eta),
        "num_selected": int(selected_indices.numel()),
    }


def get_selected_indices_zscore(
    cosine_similarities: torch.Tensor,
    *,
    tau: float = 2.5,
    max_selected: int | None = None,
):
    x = cosine_similarities
    if x.numel() == 0:
        return x.new_empty(0, dtype=torch.long), {"method": "zscore", "tau": float(tau), "num_selected": 0}
    mu = x.mean()
    sigma = x.std(unbiased=False)
    sigma_val = float(sigma.item())
    if sigma_val <= 0.0:
        return x.new_empty(0, dtype=torch.long), {
            "method": "zscore",
            "tau": float(tau),
            "mu": float(mu.item()),
            "sigma": sigma_val,
            "num_selected": 0,
            "num_pos": 0,
        }

    upper = mu + tau * sigma
    lower = mu - tau * sigma
    pos_idx = torch.nonzero(x >= upper, as_tuple=False).flatten()
    # if zero, select the max positive value
    if pos_idx.numel() == 0:
        pos_idx = torch.argmax(x)
    selected = _sort_unique_by_abs_desc(pos_idx, x)
    selected = _cap_indices(selected, max_selected=max_selected)
    return selected, {
        "method": "zscore",
        "tau": float(tau),
        "mu": float(mu.item()),
        "sigma": sigma_val,
        "threshold_upper": float(upper.item()),
        "threshold_lower": float(lower.item()),
        "num_pos": int(pos_idx.numel()),
        "num_selected": int(selected.numel()),
    }


def get_selected_indices_fdr(
    cosine_similarities: torch.Tensor,
    *,
    q0: float = 0.05,
    max_selected: int | None = None,
):
    x = cosine_similarities
    if x.numel() == 0:
        return x.new_empty(0, dtype=torch.long), {"method": "fdr", "q0": float(q0), "num_selected": 0}

    mu = x.mean()
    sigma = x.std(unbiased=False).clamp_min(1e-12)
    z = (x - mu) / sigma
    # Two-sided p-values under normal null.
    pvals = torch.erfc(z.abs() / math.sqrt(2.0)).clamp_min(1e-12).clamp_max(1.0)
    m = int(pvals.numel())
    sorted_pvals, sorted_idx = torch.sort(pvals, descending=False)
    ranks = torch.arange(1, m + 1, device=x.device, dtype=x.dtype)
    thresholds = (q0 * ranks) / float(m)
    passed = sorted_pvals <= thresholds
    if not bool(passed.any()):
        selected = x.new_empty(0, dtype=torch.long)
        p_cut = None
    else:
        k = int(torch.nonzero(passed, as_tuple=False)[-1].item()) + 1
        p_cut = float(sorted_pvals[k - 1].item())
        selected = sorted_idx[:k]

    selected = _sort_unique_by_abs_desc(selected, x)
    selected = _cap_indices(selected, max_selected=max_selected)
    return selected, {
        "method": "fdr",
        "q0": float(q0),
        "mu": float(mu.item()),
        "sigma": float(sigma.item()),
        "p_cut": p_cut,
        "num_selected": int(selected.numel()),
        "num_tests": m,
    }


def get_selected_indices_gmm(
    cosine_similarities: torch.Tensor,
    *,
    posterior_threshold: float = 0.8,
    max_selected: int | None = None,
    max_iter: int = 50,
):
    x = cosine_similarities
    if x.numel() == 0:
        return x.new_empty(0, dtype=torch.long), {"method": "gmm", "num_selected": 0}
    x1 = x.reshape(-1, 1)
    n = x1.shape[0]
    mu = torch.quantile(x, torch.tensor([0.25, 0.75], device=x.device, dtype=x.dtype)).reshape(2, 1)
    var0 = x.var(unbiased=False).clamp_min(1e-6)
    var = torch.full((2, 1), var0, device=x.device, dtype=x.dtype)
    pi = torch.full((2, 1), 0.5, device=x.device, dtype=x.dtype)
    two_pi = torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype)

    for _ in range(max_iter):
        # E-step
        norm = (1.0 / torch.sqrt(two_pi * var)) * torch.exp(-0.5 * ((x1 - mu) ** 2) / var)
        weighted = pi * norm
        denom = weighted.sum(dim=0, keepdim=True).clamp_min(1e-12)
        resp = weighted / denom
        # M-step
        Nk = resp.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pi = Nk / float(n)
        mu = (resp @ x1) / Nk
        centered = x1.unsqueeze(0) - mu.unsqueeze(1)
        var = ((resp.unsqueeze(2) * (centered**2)).sum(dim=1) / Nk).clamp_min(1e-6)

    # "Relevant" component: larger absolute mean.
    relevant_idx = int(torch.argmax(mu.abs()).item())
    posterior_relevant = resp[relevant_idx]
    selected = torch.nonzero(posterior_relevant > posterior_threshold, as_tuple=False).flatten()
    selected = _sort_unique_by_abs_desc(selected, x)
    selected = _cap_indices(selected, max_selected=max_selected)
    return selected, {
        "method": "gmm",
        "posterior_threshold": float(posterior_threshold),
        "means": [float(mu[0].item()), float(mu[1].item())],
        "stds": [float(torch.sqrt(var[0]).item()), float(torch.sqrt(var[1]).item())],
        "weights": [float(pi[0].item()), float(pi[1].item())],
        "relevant_component": relevant_idx,
        "num_selected": int(selected.numel()),
    }


def get_selected_indices_sparse_pursuit(
    cosine_similarities: torch.Tensor,
    *,
    target_k: int | None = None,
    l1_lambda: float = 0.0,
    max_selected: int | None = None,
):
    x = cosine_similarities
    if x.numel() == 0:
        return x.new_empty(0, dtype=torch.long), {"method": "sparse_pursuit", "num_selected": 0}

    if target_k is not None and target_k > 0:
        k = min(int(target_k), int(x.numel()))
        selected = torch.topk(x.abs(), k=k).indices
        mode = "topk_abs"
        threshold = float(torch.topk(x.abs(), k=k).values[-1].item()) if k > 0 else 0.0
    else:
        lam = float(l1_lambda)
        selected = torch.nonzero(x.abs() >= lam, as_tuple=False).flatten()
        mode = "l1_threshold"
        threshold = lam

    selected = _sort_unique_by_abs_desc(selected, x)
    selected = _cap_indices(selected, max_selected=max_selected)
    return selected, {
        "method": "sparse_pursuit",
        "mode": mode,
        "threshold": threshold,
        "target_k": None if target_k is None else int(target_k),
        "l1_lambda": float(l1_lambda),
        "num_selected": int(selected.numel()),
    }


def get_selected_indices_quantile(
    cosine_similarities: torch.Tensor,
    *,
    q: float = 0.05,
    positive_only: bool = True,
    max_selected: int | None = None,
):
    x = cosine_similarities
    if x.numel() == 0:
        return x.new_empty(0, dtype=torch.long), {"method": "quantile", "q": float(q), "num_selected": 0}

    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(f"quantile selection expects q in (0, 1), got {q}")

    if positive_only:
        pool = x[x > 0]
        if pool.numel() == 0:
            return x.new_empty(0, dtype=torch.long), {
                "method": "quantile",
                "q": q,
                "positive_only": True,
                "threshold": None,
                "num_selected": 0,
            }
        threshold = torch.quantile(pool, 1.0 - q)
        selected = torch.nonzero(x >= threshold, as_tuple=False).flatten()
        selected = selected[x[selected] > 0]
    else:
        threshold = torch.quantile(x.abs(), 1.0 - q)
        selected = torch.nonzero(x.abs() >= threshold, as_tuple=False).flatten()

    selected = _sort_unique_by_abs_desc(selected, x)
    selected = _cap_indices(selected, max_selected=max_selected)
    return selected, {
        "method": "quantile",
        "q": q,
        "positive_only": bool(positive_only),
        "threshold": float(threshold.item()) if threshold is not None else None,
        "num_selected": int(selected.numel()),
    }


def get_selected_indices(
    cosine_similarities: torch.Tensor,
    *,
    method: str = "abs_mass",
    **kwargs,
):
    method_key = str(method).lower()
    if method_key in {"abs_mass", "cosine_mass", "k95"}:
        return get_selected_indices_abs_mass(cosine_similarities, **kwargs)
    if method_key in {"zscore", "z_score"}:
        return get_selected_indices_zscore(cosine_similarities, **kwargs)
    if method_key in {"fdr", "bh"}:
        return get_selected_indices_fdr(cosine_similarities, **kwargs)
    if method_key in {"gmm", "mixture"}:
        return get_selected_indices_gmm(cosine_similarities, **kwargs)
    if method_key in {"sparse_pursuit", "lasso", "topk"}:
        return get_selected_indices_sparse_pursuit(cosine_similarities, **kwargs)
    if method_key in {"quantile", "q"}:
        return get_selected_indices_quantile(cosine_similarities, **kwargs)
    raise ValueError(f"Unknown selection method '{method}'.")


def attention(q, k, v):
    return torch.nn.functional.softmax(q @ k.T / math.sqrt(q.shape[-1]), dim=-1) @ v


def _attention_over_neurons(
    query: torch.Tensor,
    keys: torch.Tensor,
    tau: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Attention over neuron prototypes.

    Args
    ----
    query: [1, d] or [d]
    keys:  [K, d]
    tau: temperature (smaller => peakier)
    normalize: if True, use cosine logits via L2-normalization

    Returns
    -------
    alpha: [K] attention weights over neurons
    """
    if query.dim() == 1:
        query = query.unsqueeze(0)  # [1, d]
    assert query.dim() == 2 and keys.dim() == 2, (query.shape, keys.shape)
    assert query.shape[-1] == keys.shape[-1], (query.shape, keys.shape)

    if normalize:
        q = F.normalize(query, dim=-1)          # [1, d]
        k = F.normalize(keys, dim=-1)           # [K, d]
        logits = (q @ k.t()).squeeze(0)         # [K], cosine logits
    else:
        # scaled dot-product in raw space
        logits = (query @ keys.t()).squeeze(0) / math.sqrt(query.shape[-1])  # [K]

    alpha = F.softmax(logits / tau, dim=-1)     # [K]
    return alpha


def _raw_cosine_for_concept(
    *,
    model,
    concept: str,
    device: torch.device,
    text_encoder=None,
    basis_chunk_size: int = 256,
    attention: bool = False,
    attention_tau: float = 0.07,
    attention_normalize: bool = True,
    attention_score_mode: str = "alpha",
):
    """Compute raw cosine-like scores for one concept with optional attention reweighting."""
    encoder = text_encoder if text_encoder is not None else getattr(model, "text_encoder", None)
    if encoder is None:
        raise ValueError("A text encoder must be provided when model.text_encoder is not available.")

    text_embedding = encoder([concept])  # [1, d]
    sae_text_out = model.sae_encoder(text_embedding)
    if isinstance(sae_text_out, (tuple, list)):
        sae_text_activations = sae_text_out[1] if len(sae_text_out) > 1 else sae_text_out[0]
    else:
        sae_text_activations = sae_text_out

    dict_size = int(model.sae_decoder.W_dec.shape[0])
    if basis_chunk_size <= 0:
        basis_chunk_size = dict_size

    basis = []
    text_vec = text_embedding
    for start in range(0, dict_size, basis_chunk_size):
        end = min(start + basis_chunk_size, dict_size)
        chunk = end - start
        eye_chunk = torch.zeros((chunk, dict_size), device=device, dtype=text_vec.dtype)
        eye_chunk[torch.arange(chunk, device=device), torch.arange(start, end, device=device)] = 1
        basis_chunk = model.sae_decoder(eye_chunk)
        basis.append(basis_chunk)
    basis = torch.cat(basis, dim=0)  # [K, d]

    if attention:
        alpha = _attention_over_neurons(
            query=text_vec, keys=basis, tau=attention_tau, normalize=attention_normalize
        )
        cosine_similarities = F.cosine_similarity(text_vec.expand_as(basis), basis, dim=-1)
        if attention_score_mode == "alpha":
            cosine_similarities = alpha
        elif attention_score_mode == "alpha_cos":
            cosine_similarities = alpha * cosine_similarities
        elif attention_score_mode == "cos":
            pass
        else:
            raise ValueError(
                f"Unknown attention_score_mode={attention_score_mode}. "
                "Use one of {'alpha','alpha_cos','cos'}."
            )
    else:
        cosine_similarities = F.cosine_similarity(text_vec.expand_as(basis), basis, dim=-1)

    return cosine_similarities, sae_text_activations


def _cosine_probe_vector_and_stats(
    *,
    model,
    concept: str,
    concept_index: int,
    device: torch.device,
    selection_method: str = "abs_mass",
    selection_kwargs: Dict[str, Any] | None = None,
    basis_chunk_size: int = 256,
    attention: bool = False,
    attention_tau: float = 0.07,
    attention_normalize: bool = True,
    attention_score_mode: str = "alpha",  # {"alpha", "alpha_cos", "cos"}
):
    del concept_index  # Kept for extractor signature consistency.
    cosine_similarities, sae_text_activations = _raw_cosine_for_concept(
        model=model,
        concept=concept,
        device=device,
        basis_chunk_size=basis_chunk_size,
        attention=attention,
        attention_tau=attention_tau,
        attention_normalize=attention_normalize,
        attention_score_mode=attention_score_mode,
    )

    # --- Selection (unchanged) ---
    selection_kwargs = dict(selection_kwargs or {})

    selected_indices, selection_stats = get_selected_indices(
        cosine_similarities,
        method=selection_method,
        **selection_kwargs,
    )

    mask = torch.zeros_like(cosine_similarities)
    if selected_indices.numel() > 0:
        mask[selected_indices] = 1
    masked_cosine = cosine_similarities * mask

    probs = torch.softmax(cosine_similarities.abs(), dim=0)
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    stats = dict(selection_stats)
    stats["entropy"] = entropy

    return {
        "raw_cosine_similarities": cosine_similarities,
        "masked_cosine": masked_cosine,
        "sae_text_activations": sae_text_activations,
        "stats": stats,
    }


@torch.no_grad()
def cosine_with_tfidf(
    *,
    model,
    concepts: list[str],
    device: torch.device,
    tau: float = 4.0,
    basis_chunk_size: int = 256,
    attention: bool = False,
    attention_tau: float = 0.07,
    attention_normalize: bool = True,
    attention_score_mode: str = "alpha",
):
    """
    Two-pass cosine extraction with global TF-IDF neuron reweighting.

    Steps:
      1) raw cosine per concept
      2) z-score mask on raw cosine
      3) compute global neuron TF-IDF weights from first-pass masks
      4) apply weights to raw cosine
      5) z-score mask again on weighted cosine
    """
    if not concepts:
        return {}

    raw_by_concept: Dict[str, torch.Tensor] = {}
    sae_by_concept: Dict[str, torch.Tensor] = {}
    first_mask_by_concept: Dict[str, torch.Tensor] = {}

    # Pass 1: raw cosine + first z-score mask
    for concept in concepts:
        raw_cos, sae_text_acts = _raw_cosine_for_concept(
            model=model,
            concept=concept,
            device=device,
            basis_chunk_size=basis_chunk_size,
            attention=attention,
            attention_tau=attention_tau,
            attention_normalize=attention_normalize,
            attention_score_mode=attention_score_mode,
        )
        idx1, stats1 = get_selected_indices_zscore(raw_cos, tau=tau)
        mask1 = torch.zeros_like(raw_cos)
        if idx1.numel() > 0:
            mask1[idx1] = 1
        raw_by_concept[concept] = raw_cos
        sae_by_concept[concept] = sae_text_acts
        first_mask_by_concept[concept] = mask1

    # Global TF-IDF weights over neurons from first-pass masks.
    first_mask_matrix = torch.stack([first_mask_by_concept[c] for c in concepts], dim=0)  # [C, K]
    raw_matrix = torch.stack([raw_by_concept[c] for c in concepts], dim=0)  # [C, K]
    n_concepts = first_mask_matrix.shape[0]
    doc_freq = first_mask_matrix.sum(dim=0)  # [K]
    tf = (raw_matrix.abs() * first_mask_matrix).sum(dim=0) / max(n_concepts, 1)  # [K]
    idf = torch.log((n_concepts + 1.0) / (doc_freq + 1.0)) + 1.0  # [K]
    neuron_weights = tf * idf
    # Normalize to keep scales comparable across runs.
    neuron_weights = neuron_weights / neuron_weights.mean().clamp_min(1e-12)

    out = {}
    # Pass 2: reweight raw cosine + second z-score mask
    for concept in concepts:
        raw_cos = raw_by_concept[concept]
        weighted_cos = raw_cos * neuron_weights
        idx2, stats2 = get_selected_indices_zscore(weighted_cos, tau=tau)
        mask2 = torch.zeros_like(weighted_cos)
        if idx2.numel() > 0:
            mask2[idx2] = 1
        masked_weighted = weighted_cos * mask2

        probs = torch.softmax(weighted_cos.abs(), dim=0)
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
        stats = dict(stats2)
        stats["method"] = "cosine_with_tfidf"
        stats["entropy"] = entropy
        stats["tau"] = float(tau)
        stats["tfidf_weight_mean"] = float(neuron_weights.mean().item())
        stats["tfidf_weight_max"] = float(neuron_weights.max().item())
        stats["initial_num_selected"] = int(first_mask_by_concept[concept].sum().item())

        out[concept] = {
            # Save weighted cosine as the primary "raw_cosine_similarities" output
            # for downstream compatibility with existing plotting scripts.
            "raw_cosine_similarities": weighted_cos,
            # Keep original pre-TF-IDF cosine values for debugging/ablation.
            "original_raw_cosine_similarities": raw_cos,
            "weighted_cosine_similarities": weighted_cos,
            "masked_cosine": masked_weighted,
            "sae_text_activations": sae_by_concept[concept],
            "stats": stats,
        }

    return out

def build_concept_extractor(extractor_cfg: DictConfig):
    """Resolve extractor callable from config and apply partial kwargs."""
    if extractor_cfg is None:
        raise ValueError("Missing concepts_experiment extractor config.")

    target = extractor_cfg.get("target")
    if target is None:
        # Backward compatibility with older config format.
        target = extractor_cfg.get("_extract_concept_distribution")
    if not target:
        raise ValueError(
            "Extractor target missing. Set concepts_experiment.extractor.target "
            "(or legacy methods._extract_concept_distribution)."
        )

    module_name, fn_name = str(target).rsplit(".", 1)
    fn = getattr(importlib.import_module(module_name), fn_name)
    if not callable(fn):
        raise TypeError(f"Resolved extractor is not callable: {target}")

    params_cfg = extractor_cfg.get("params", {})
    params = dict(params_cfg) if params_cfg is not None else {}
    return partial(fn, **params)



def _assert_checkpoint_weights_loaded(model) -> None:
    report = getattr(model, "_checkpoint_load_report", None)
    if report is None:
        raise RuntimeError(
            "Checkpoint load report missing on model. "
            "Ensure model is created via instantiate_model_and_load()."
        )
    if int(report.get("matched_keys", 0)) <= 0:
        raise RuntimeError("Checkpoint weights do not match model parameters (0 matched keys).")
    log.info(
        "Checkpoint load verification passed: "
        f"matched={report['matched_keys']}/{report['model_keys']}, "
        f"missing={report['missing_keys']}, unexpected={report['unexpected_keys']}."
    )


def concept_from_bridge_score(
    *,
    model,
    concept: str,
    concept_index: int,
    device: torch.device,
    bridge_checkpoint_path: str = None,
    bridge_checkpoint_key: str = "bridges",
    dataloader=None,
) -> torch.Tensor:
    del concept_index  # kept for a stable extractor callable signature

    def _extract_bridge_from_checkpoint(path: str, key: str):
        checkpoint = load_state_dict_any(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            return None

        bridge_obj = state_dict.get(key)
        if bridge_obj is None:
            return None

        if isinstance(bridge_obj, torch.Tensor):
            return bridge_obj

        if isinstance(bridge_obj, dict):
            bridge_tensors = []
            for _, val in sorted(bridge_obj.items(), key=lambda kv: str(kv[0])):
                # New structure: bridges[dataset_name] = {"align": ..., "coact": ..., "B": ...}
                if isinstance(val, dict):
                    b_val = val.get("B")
                    if isinstance(b_val, torch.Tensor):
                        bridge_tensors.append(b_val)
                # Fallback: dict values are directly bridge tensors.
                elif isinstance(val, torch.Tensor):
                    bridge_tensors.append(val)
            if bridge_tensors:
                if len(bridge_tensors) == 1:
                    return bridge_tensors[0]
                return torch.stack(bridge_tensors, dim=0).mean(dim=0)
        return None

    @torch.no_grad()
    def _compute_bridge_from_dataloader(dataloader_obj) -> torch.Tensor:
        za_chunks = []
        zt_chunks = []
        for batch in dataloader_obj:
            audio = batch.get("audio") if isinstance(batch, dict) else None
            text = batch.get("prompt") if isinstance(batch, dict) else None
            if audio is None or text is None:
                continue

            audio = audio.to(device)
            encoded_audio = (
                model.audio_encoder(audio) if not getattr(model, "preextracted_features", False) else audio
            )
            encoded_text = model.text_encoder(text)
            _, z_audio, _, _ = model(encoded_audio)
            _, z_text, _, _ = model(encoded_text)
            n = min(z_audio.shape[0], z_text.shape[0])
            if n <= 0:
                continue
            za_chunks.append(z_audio[:n].detach())
            zt_chunks.append(z_text[:n].detach())

        if not za_chunks or not zt_chunks:
            raise ValueError(
                "Could not compute bridge from dataloader: no batches contained both audio and prompt."
            )

        za = torch.cat(za_chunks, dim=0)
        zt = torch.cat(zt_chunks, dim=0)
        w_dec = get_dictionary_from_lightningsae(model).to(device)
        return BridgeScoreCallback._compute(za, zt, w_dec)["B"]

    bridge_B = None
    if bridge_checkpoint_path:
        bridge_B = _extract_bridge_from_checkpoint(bridge_checkpoint_path, bridge_checkpoint_key)

    if bridge_B is None:
        if dataloader is None:
            raise ValueError(
                "bridges not found in checkpoint (or checkpoint not provided). "
                "Provide dataloader to compute bridge scores on the fly."
            )
        bridge_B = _compute_bridge_from_dataloader(dataloader)

    bridge_B = bridge_B.to(device)
    text_embedding = model.text_encoder([concept])
    activations = model.sae_encoder(text_embedding)
    bridged_activations = bridge_B @ activations
    return bridged_activations.detach().cpu()


def run_extract_experiment(
    *,
    cfg: DictConfig,
    model,
    device: torch.device,
    output_dir: str,
    source_run_id: str,
    ckpt_path: str,
) -> Dict[str, Any]:


    extractor_cfg = cfg.concepts_experiment.get("extractor")
    if extractor_cfg is None:
        extractor_cfg = cfg.concepts_experiment.get("methods", {})


    extractor = build_concept_extractor(extractor_cfg)
    extractor_target = str(extractor_cfg.get("target", ""))
    concepts = read_vocab_lines(str(cfg.concepts.vocab_path))
    _assert_checkpoint_weights_loaded(model)

    out_ = {}

    if extractor_target.endswith("cosine_with_tfidf"):
        # Batch method needs all concepts to compute global neuron TF-IDF weights.
        tfidf_results = extractor(
            model=model,
            concepts=concepts,
            device=device,
        )
        for concept in concepts:
            results = tfidf_results[concept]
            masked_cosine = results["masked_cosine"].detach().cpu()
            raw_cosine = results["raw_cosine_similarities"].detach().cpu()
            weighted_cosine = results.get("weighted_cosine_similarities")
            if isinstance(weighted_cosine, torch.Tensor):
                weighted_cosine = weighted_cosine.detach().cpu()
            sae_acts = results["sae_text_activations"]
            if isinstance(sae_acts, torch.Tensor):
                sae_acts = sae_acts.detach().cpu()
            out_[concept] = {
                "masked_cosine": masked_cosine,
                "stats": results["stats"],
                "raw_cosine_similarities": raw_cosine,
                "weighted_cosine_similarities": weighted_cosine,
                "sae_text_activations": sae_acts,
            }
    else:
        for idx, concept in enumerate(concepts):
            results = extractor(
                model=model,
                concept=concept,
                concept_index=idx,
                device=device,
            )
            masked_cosine = results["masked_cosine"].detach().cpu()
            raw_cosine = results["raw_cosine_similarities"].detach().cpu()
            sae_acts = results["sae_text_activations"]
            if isinstance(sae_acts, torch.Tensor):
                sae_acts = sae_acts.detach().cpu()
            out_[concept] = {
                "masked_cosine": masked_cosine,
                "stats": results["stats"],
                "raw_cosine_similarities": raw_cosine,
                "sae_text_activations": sae_acts,
            }

    # Print a compact activation summary for quick inspection.
    # "Active" means non-zero entries in the extracted activation tensor.
    width = 40
    topk_width = 20
    summary_rows = []
    total_active = 0
    total_values = 0
    global_max = None
    for concept, values in out_.items():
        tensor = values['masked_cosine'].detach().cpu()
        if tensor.numel() == 0:
            active = 0
            size = 0
            frac = 0.0
            max_val = 0.0
            topk_pairs = []
        else:
            active = int((tensor != 0).sum().item())
            size = int(tensor.numel())
            frac = active / max(size, 1)
            max_val = float(tensor.max().item())
            flat = tensor.flatten()
            k = min(5, flat.numel())
            top_vals = torch.topk(flat, k=k).values
            denom = max(abs(max_val), 1e-12)
            topk_pairs = []
            for v in top_vals:
                v_float = float(v.item())
                rel = max(0.0, min(1.0, v_float / denom))
                filled_top = int(round(rel * topk_width))
                top_bar = "#" * filled_top + "-" * (topk_width - filled_top)
                topk_pairs.append((v_float, top_bar))
        if global_max is None:
            global_max = max_val
        else:
            global_max = max(global_max, max_val)
        total_active += active
        total_values += size
        filled = int(round(frac * width))
        bar = "#" * filled + "-" * (width - filled)
        summary_rows.append((concept, active, size, frac, max_val, bar, topk_pairs))

    log.info("Activation summary per concept (active/total | active_frac | max):")
    for concept, active, size, frac, max_val, bar, topk_pairs in summary_rows:
        log.info(
            f"- {concept:>24s} | {active:>7d}/{size:<7d} | {frac:6.2%} | "
            f"max={max_val:>9.4f} | [{bar}]"
        )
        if topk_pairs:
            topk_str = " | ".join([f"{val:9.4f} [{top_bar}]" for val, top_bar in topk_pairs])
            log.info(f"  top5: {topk_str}")
        else:
            log.info("  top5: n/a")
    global_frac = (total_active / total_values) if total_values > 0 else 0.0
    if global_max is None:
        global_max = 0.0
    log.info(
        "Activation summary total: "
        f"{total_active}/{total_values} active ({global_frac:.2%}), global max={global_max:.4f}."
    )
    if out_:
        for concept, values in out_.items():
            stats = values['stats']
            k95 = int(stats.get("k95", 0))
            entropy = float(stats.get("entropy", 0.0))
            log.info(
                f"{concept:24s} | {k95:4d} | {entropy:9.4f}"
            )


    # Resolve output directory: <output_dir>/<run_id>/<experiment_id>
    output_dir, run_id_slug, experiment_id = _resolve_output_dir(cfg, output_dir, source_run_id)

    # Save activations under concepts.pt
    activations_path = os.path.join(output_dir, "concepts.pt")
    torch.save(out_, activations_path)
    log.info(f"Saved concept activation dictionary to {activations_path}")

    # Save config
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)


    log.info(f"Saved concept activation dictionary to {activations_path}")

    out_dict = {
        "experiment": "concept_isolation",
        "concepts_experiment": str(cfg.concepts_experiment.name),
        "config_source": source_run_id,
        "run_id": run_id_slug,
        "experiment_id": experiment_id,
        "checkpoint_path": ckpt_path,
        "num_concepts": len(concepts),
        "concept_activations_path": activations_path,
    }

    # Save JSON summary output
    json_output_path = os.path.join(output_dir, "output.json")
    import json
    with open(json_output_path, "w") as f:
        json.dump(out_dict, f)

    return out_dict
