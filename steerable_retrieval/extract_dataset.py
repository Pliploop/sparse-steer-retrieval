import os
from typing import Any, Dict, Optional, Set
from datetime import datetime

import lightning as L
import rootutils
import torch
import hydra
from omegaconf import DictConfig, OmegaConf


from dora import get_xp, hydra_main
import hydra

import lightning as L
import rootutils
import torch
from omegaconf import DictConfig


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from gdr import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from steerable_retrieval.utils import (
    RankedLogger,
    extras,
    register_resolvers,
)

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()


def list_existing_paths(path: str) -> Set[str]:
    """List existing feature files from either a local directory or S3 path.
    
    Args:
        path: Either a local directory path or S3 path (s3://bucket/path) where features are stored
        
    Returns:
        Set of relative file paths (done IDs) extracted from existing files
    """
    done_ids = set()
    
    if not path:
        log.info("No path provided, skipping path listing")
        return done_ids
    
    # Check if it's an S3 path
    if 's3://' in path:
        # S3 path - list from S3
        try:
            import boto3
            client = boto3.client('s3')
            
            # Extract bucket and prefix from S3 path
            s3_path = path.replace("s3://", "")
            parts = s3_path.split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            if prefix and not prefix.endswith('/'):
                prefix = prefix + '/'
            
            log.info(f"Listing existing paths from S3: s3://{bucket}/{prefix}")
            
            # Paginate through all objects in the target location
            paginator = client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            path_count = 0
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        object_key = obj['Key']
                        # Extract relative path by removing the prefix
                        if object_key.startswith(prefix):
                            relative_path = object_key[len(prefix):]
                            # Skip empty paths and directories (ending with /)
                            if relative_path and not relative_path.endswith('/'):
                                # Add the path as-is
                                done_ids.add(relative_path.replace('.npy','').replace('.wav','').replace('.mp3',''))
                                # remove extension
                                path_count += 1
            
            log.info(f"Found {path_count} existing paths in S3, {len(done_ids)} unique done IDs")
            
        except Exception as e:
            log.warning(f"Failed to list existing paths from S3: {e}")
            log.warning("Continuing without skipping already processed items")
    else:
        # Local directory - list from filesystem
        try:
            if not os.path.exists(path):
                log.info(f"Local path does not exist: {path}, skipping path listing")
                return done_ids
            
            log.info(f"Listing existing paths from local directory: {path}")
            
            # Walk through the directory and collect all .npy files
            path_count = 0
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.npy'):
                        # Get relative path from the base path
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, path)
                        # Normalize path separators (use forward slashes)
                        relative_path = relative_path.replace(os.sep, '/')
                        
                        # Add the path as-is
                        done_ids.add(relative_path.replace('.npy','').replace('.wav','').replace('.mp3',''))
                        path_count += 1
            
            log.info(f"Found {path_count} existing paths locally, {len(done_ids)} unique done IDs")
            
        except Exception as e:
            log.warning(f"Failed to list existing paths from local directory: {e}")
            log.warning("Continuing without skipping already processed items")
    
    return done_ids


def extract_features(cfg: DictConfig) -> Dict[str, Any]:
    """Extracts features from audio datasets using pre-trained encoders.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A dict with extraction metadata.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # Setup datamodule
    datamodule.setup(None)
    
    # Move encoder_pair to device
    device = cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    model.to(device)
    log.info(f"Using device: {device}")


    # Get extract parameters from config
    save_dir = cfg.get("save_dir")
    root_path = cfg.get("root_path")
    extract_method = cfg.get("extract_method", "get_audio_embedding_from_data")
    out_key = cfg.get("out_key", None)
    hop = cfg.get("hop", 48000)
    limit_n = cfg.get("limit_n")
    save = cfg.get("save", False)
    sagemaker_dir = cfg.get("sagemaker_dir", None)

    # Handle save_dir
    if save_dir is None:
        if datamodule.train_dataset is not None and len(datamodule.train_dataset.annotations) > 0:
            save_dir = os.path.dirname(datamodule.train_dataset.annotations[0]['file_path'])
        else:
            raise ValueError("save_dir must be specified in config or available from dataset")

    # Check for sagemaker config to get S3 output destination
    # In sagemaker, files are saved locally to save_dir, then uploaded to S3 destination
    s3_destination = sagemaker_dir
    log.info(f"S3 destination: {s3_destination}")
    # List existing paths on restart to build done_ids set
    # Check both save_dir and s3_destination (if using SageMaker) to avoid re-extracting
    done_ids = set()
    if save:
        # Check save_dir (local or S3)
        done_ids_save_dir = list_existing_paths(save_dir)
        if done_ids_save_dir:
            log.info(f"Found {len(done_ids_save_dir)} existing files in save_dir: {save_dir}")
        done_ids.update(done_ids_save_dir)
        
        # Also check S3 destination if using SageMaker
        if s3_destination:
            done_ids_s3 = list_existing_paths(s3_destination)
            if done_ids_s3:
                log.info(f"Found {len(done_ids_s3)} existing files in S3 destination: {s3_destination}")
            done_ids.update(done_ids_s3)
        
        if done_ids:
            log.info(f"Total: Skipping {len(done_ids)} already processed items")
        else:
            log.info("No existing paths found in save_dir or S3 destination")


    log.info(f"Extracting features with {extract_method} method")
    log.info(f"Saving to: {save_dir}")
    log.info(f"out_key: {out_key}, hop: {hop}, limit_n: {limit_n}, save: {save}")

    # Save config to save_dir
    if 's3://' in save_dir:
        # S3 path - upload config
        try:
            import s3fs
            fs = s3fs.S3FileSystem()
            config_path = f"{save_dir}/config.yaml"
            
            with fs.open(config_path, "w") as config_file:
                config_file.write(OmegaConf.to_yaml(cfg, resolve=True))
            log.info(f"Uploaded config to {config_path}")
        except Exception as e:
            log.warning(f"Failed to upload config to S3: {e}")
    else:
        # Local path - save config
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as config_file:
            config_file.write(OmegaConf.to_yaml(cfg, resolve=True))
        log.info(f"Saved config to {config_path}")

    # Extract features from all datasets
    datasets = []
    if datamodule.train_dataset is not None:
        datasets.append(datamodule.train_dataset)
    if datamodule.val_datasets:
        datasets.extend([d for d in datamodule.val_datasets if d is not None])
    if datamodule.test_datasets:
        datasets.extend([d for d in datamodule.test_datasets if d is not None])

    for dataset in datasets:
        if dataset is not None:
            log.info(f"Extracting features from dataset: {type(dataset).__name__}")
            dataset.extract_and_save_features(
                model.audio_encoder,
                save_dir=save_dir,
                extract_method=extract_method,
                out_key=out_key,
                hop=hop,
                limit_n=limit_n,
                save=save,
                verbose=False,
                root_path=root_path,
                done_ids=done_ids
            )

    log.info("Feature extraction completed!")
    
    return {
        "save_dir": save_dir,
        "extract_method": extract_method,
        "out_key": out_key,
    }


@hydra_main(version_base="1.3", config_path="../configs", config_name="extract_feature.yaml")
def main(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Main entry point for feature extraction.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Dict with extraction metadata.
    """
    # handle A100 GPUs
    if torch.cuda.is_available() and ("A100" in torch.cuda.get_device_name() or "A5000" in torch.cuda.get_device_name()):
        torch.set_float32_matmul_precision("high")

    # avoid annoying multiprocessing errors
    torch.multiprocessing.set_sharing_strategy('file_system')

    # prevent annoying warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # extract features
    result = extract_features(cfg)

    return result


if __name__ == "__main__":
    main()

