# Offline Processing Experiments

This directory contains offline experiment jobs that run outside training callbacks.

## Stability (offline)

Execution script:
- `python steerable_retrieval/experiments/run_stability.py model_source.config_source=<WANDB_RUN_ID_OR_YAML_PATH> model_source.checkpoint_source=<CKPT> stability.checkpoint_paths=[<CKPT1>,<CKPT2>]`

Launch as SageMaker processing job:
- `python launch_stability.py model_source.config_source=<WANDB_RUN_ID_OR_YAML_PATH> model_source.checkpoint_source=<CKPT> stability.checkpoint_paths=[<CKPT1>,<CKPT2>]`

## Concept Isolation (offline)

Execution script:
- `python steerable_retrieval/experiments/run_concept_isolation.py model_source.config_source=<WANDB_RUN_ID_OR_YAML_PATH> model_source.checkpoint_source=<CKPT> concepts_experiment=extract concepts.vocab_path=<VOCAB_TXT>`

Score edit (add):
- `python steerable_retrieval/experiments/run_concept_isolation.py model_source.config_source=<WANDB_RUN_ID_OR_YAML_PATH> model_source.checkpoint_source=<CKPT> concepts_experiment=score_edit_add concepts.vocab_path=<VOCAB_TXT>`

Score edit (suppress):
- `python steerable_retrieval/experiments/run_concept_isolation.py model_source.config_source=<WANDB_RUN_ID_OR_YAML_PATH> model_source.checkpoint_source=<CKPT> concepts_experiment=score_edit_suppress concepts.vocab_path=<VOCAB_TXT>`

Launch as SageMaker processing job:
- `python launch_concept_isolation.py model_source.config_source=<WANDB_RUN_ID_OR_YAML_PATH> model_source.checkpoint_source=<CKPT> concepts_experiment=extract concepts.vocab_path=<VOCAB_TXT>`

## Notes

- Both jobs save run config and outputs under `output_dir` (default `${paths.output_dir}`).
- Stability logging to W&B is controlled by `wandb.*` keys in `configs/experiment/stability.yaml`.
- Concept isolation is now a dispatcher with `concepts_experiment=extract|score_edit_add|score_edit_suppress`.
- `model_source.config_source` can be a W&B run id, a local `.yaml` file path, or an `s3://...yaml` path.
- If `config_source` contains `.yaml`, the model config is read from that YAML; otherwise it is fetched from W&B.
- `model_source.checkpoint_source` is the checkpoint path loaded directly (local path or S3 path).

