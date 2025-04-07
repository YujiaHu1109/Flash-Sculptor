from omegaconf import OmegaConf

def load_cfg(cfg_path):
    return OmegaConf.load(cfg_path)

def merge_cfgs(cfg1,cfg2):
    cfg = OmegaConf.merge(cfg1,cfg2)
    return cfg