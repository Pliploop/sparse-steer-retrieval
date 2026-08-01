import copy
import math
import os
from typing import Any, Dict, List, Optional

import lightning as L
import rootutils
import torch
import torch.nn.functional as F
from dora.hydra import hydra_main
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from steerable_retrieval.callbacks.energy import get_dictionary_from_lightningsae, load_state_dict_any  # noqa: E402
from steerable_retrieval.experiments.common import (  # noqa: E402
    ensure_dir,
    finish_wandb,
    instantiate_model_and_load,
    maybe_start_wandb,
    resolve_model_from_source,
    save_config,
    save_json,
)
from steerable_retrieval.utils import RankedLogger, extras, register_resolvers  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()


def _stability_hungarian(a: torch.Tensor, b: torch.Tensor) -> float:
    from scipy.optimize import linear_sum_assignment

    sim = (a @ b.t()).detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(-sim)
    return float(sim[row_idx, col_idx].mean())


def _k_values(n_concepts: int, n_curve_points: int) -> List[int]:
    if n_concepts <= 1:
        return [1]
    out = [1]
    for i in range(max(2, n_curve_points)):
        frac = i / max(n_curve_points - 1, 1)
        k = int(round(math.exp(math.log(1) * (1 - frac) + math.log(n_concepts) * frac)))
        out.append(max(1, min(n_concepts, k)))
    out.append(n_concepts)
    return sorted(set(out))


def run_stability(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    source_run_id = resolve_model_from_source(cfg)
    output_dir = ensure_dir(str(cfg.output_dir))
    save_config(cfg, output_dir)

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model, ckpt_path = instantiate_model_and_load(cfg, device=device)
    d0 = F.normalize(get_dictionary_from_lightningsae(model).to(device), dim=-1)
    n_concepts = int(d0.shape[0])

    checkpoints = list(cfg.stability.checkpoint_paths)
    if not checkpoints:
        raise ValueError("stability.checkpoint_paths cannot be empty.")

    run = maybe_start_wandb(
        cfg=cfg,
        job_type="stability_processing",
        name=str(cfg.get("run_name", "stability_processing")),
        output_dir=output_dir,
    )

    sorted_indices = torch.arange(n_concepts, device=device)
    k_values = _k_values(n_concepts=n_concepts, n_curve_points=int(cfg.stability.n_curve_points))
    topk = cfg.stability.get("topk_by_energy")

    per_checkpoint = []
    all_stabilities = []
    all_topk = []
    curves = []

    for path in checkpoints:
        comparison = copy.deepcopy(model).to("cpu")
        comparison.load_state_dict(load_state_dict_any(path, map_location="cpu"), strict=False)
        comparison = comparison.to(device).eval()
        dk = F.normalize(get_dictionary_from_lightningsae(comparison).to(device), dim=-1)

        stab_full = _stability_hungarian(d0, dk)
        all_stabilities.append(stab_full)

        stab_topk = None
        if topk is not None:
            k = min(int(topk), n_concepts)
            top_idx = sorted_indices[:k]
            stab_topk = _stability_hungarian(d0[top_idx], dk[top_idx])
            all_topk.append(stab_topk)

        curve_vals = []
        for k in k_values:
            top_idx = sorted_indices[:k]
            curve_vals.append(_stability_hungarian(d0[top_idx], dk[top_idx]))
        curves.append(curve_vals)

        per_checkpoint.append(
            {
                "checkpoint_path": str(path),
                "stability_full": stab_full,
                "stability_topk": stab_topk,
            }
        )

    mean_curve = [sum(vals) / len(vals) for vals in zip(*curves)]
    summary = {
        "experiment": "stability_processing",
        "config_source_wandb_run_id": source_run_id,
        "source_checkpoint": ckpt_path,
        "n_concepts": n_concepts,
        "per_checkpoint": per_checkpoint,
        "mean_stability_full": float(sum(all_stabilities) / len(all_stabilities)),
        "mean_stability_topk": float(sum(all_topk) / len(all_topk)) if all_topk else None,
        "k_values": k_values,
        "mean_stability_curve": mean_curve,
    }
    save_json(os.path.join(output_dir, "summary.json"), summary)

    if run is not None:
        import wandb

        table = wandb.Table(columns=["checkpoint_path", "stability_full", "stability_topk"])
        for item in per_checkpoint:
            table.add_data(item["checkpoint_path"], item["stability_full"], item["stability_topk"])
        wandb.log({"stability/per_checkpoint": table})
        wandb.log({"stability/mean_full": summary["mean_stability_full"]})
        if summary["mean_stability_topk"] is not None:
            wandb.log({"stability/mean_topk": summary["mean_stability_topk"]})

    finish_wandb(run)
    return summary


@hydra_main(version_base="1.3", config_path="../../configs/experiment", config_name="stability.yaml")
def main(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    extras(cfg)
    return run_stability(cfg)


if __name__ == "__main__":
    main()

