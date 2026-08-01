from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from steerable_retrieval.utils import pylogger

from hydra.utils import instantiate as hydra_instantiate

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

CLASS_FLAG = '_target_'

def unit_instantiate(cfg):
    if isinstance(cfg, dict):
        return hydra_instantiate(cfg)
    else:
        return cfg

def instantiate(cfg):
    # If not a dict or doesn't have the class flag, use unit_instantiate (return as-is or recursively go through lists)
    if not isinstance(cfg, dict) or CLASS_FLAG not in cfg:
        if isinstance(cfg, dict):
            # Recursively instantiate dict values, but respect _partial_ flag
            instantiated_cfg = {}
            for k, v in cfg.items():
                if isinstance(v, dict):
                    # Check if this dict has _partial_ flag - if so, don't instantiate it recursively
                    if v.get('_partial_', False):
                        instantiated_cfg[k] = v  # Keep as dict for later partial instantiation
                    else:
                        instantiated_cfg[k] = instantiate(v)
                elif isinstance(v, (list, tuple)):
                    instantiated_cfg[k] = type(v)(instantiate(x) if isinstance(x, dict) else x for x in v)
                else:
                    instantiated_cfg[k] = v
            return instantiated_cfg
        elif isinstance(cfg, (list, tuple)):
            return type(cfg)(instantiate(x) for x in cfg)
        else:
            return cfg

    # Check if this config has _partial_ flag
    if cfg.get('_partial_', False):
        # Use Hydra's instantiate with _partial_ to create a functools.partial
        return hydra_instantiate(cfg)

    instantiated_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            # Check if this nested dict has _partial_ flag
            if v.get('_partial_', False):
                instantiated_cfg[k] = v  # Keep as dict for later partial instantiation
            else:
                instantiated_cfg[k] = instantiate(v)
        elif isinstance(v, (list, tuple)):
            instantiated_cfg[k] = type(v)(instantiate(x) if isinstance(x, dict) else x for x in v)
        else:
            instantiated_cfg[k] = v
    return unit_instantiate(instantiated_cfg)

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        cb = instantiate(cb_conf)
        callbacks.append(cb) if cb is not None else None

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        # Skip partial configs that carry no _target_ (e.g. an experiment's
        # `logger.wandb:` augmentation block left dangling when the logger group
        # is overridden to something else). instantiate() would otherwise return
        # the raw DictConfig and it would masquerade as a logger downstream.
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            lg = instantiate(lg_conf)
            logger.append(lg) if lg is not None else None

    return logger
