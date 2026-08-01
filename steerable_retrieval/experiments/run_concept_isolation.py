import json
import os
from typing import Any, Dict, Optional

import lightning as L
import rootutils
import torch
from dora.hydra import hydra_main
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from steerable_retrieval.experiments.common import (  # noqa: E402
    ensure_dir,
    instantiate_model_and_load,
    resolve_model_from_source,
    save_config,
)
from steerable_retrieval.experiments.concepts.extract import run_extract_experiment
from steerable_retrieval.utils import RankedLogger, extras, register_resolvers  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()


def run_concept_isolation(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    source_run_id = resolve_model_from_source(cfg)
    output_dir = ensure_dir(str(cfg.output_dir))
    save_config(cfg, output_dir)

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model, ckpt_path = instantiate_model_and_load(cfg, device=device)
    experiment_name = str(cfg.concepts_experiment.get("name", "extract"))

    if experiment_name == "extract":
        out_dict = run_extract_experiment(
            cfg=cfg,
            model=model,
            device=device,
            output_dir=output_dir,
            source_run_id=source_run_id,
            ckpt_path=ckpt_path,
        )
    else:
        raise ValueError(
            f"Unsupported concepts_experiment.name={experiment_name}. "
            "Expected one of: extract, score_edit_add, score_edit_suppress."
        )

    # Save metadata for downstream launch tooling.
    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(out_dict, f)

    return out_dict


@hydra_main(version_base="1.3", config_path="../../configs", config_name="experiment/concept_isolation.yaml")
def main(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    extras(cfg)
    return run_concept_isolation(cfg)


if __name__ == "__main__":
    main()

