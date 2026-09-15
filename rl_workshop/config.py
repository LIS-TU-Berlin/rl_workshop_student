import omegaconf
from omegaconf import DictConfig


def load_cfg(config_path: str) -> DictConfig:
    """Loads a config yaml. If a base exists, merges it on top of that base config first.
    Variant configs only need to specify the few keys they override."""
    cfg = omegaconf.OmegaConf.load(config_path)
    if 'base' in cfg:
        base_cfg = load_cfg(cfg.base)
        del cfg['base']
        cfg = omegaconf.OmegaConf.merge(base_cfg, cfg)
    return cfg

def tag_suffix(cfg: DictConfig, last_part: str) -> str:
    """Builds a run-tag suffix like '-disc_push-task0-sbTD3', appending '-seed<N>' if cfg.seed is set."""
    task = cfg.get('task', None) or 'default'
    suffix = f'-{cfg.scenario}-{task}-{last_part}'
    seed = cfg.get('seed', None)
    if seed is not None:
        suffix += f'-seed{seed}'
    return suffix
