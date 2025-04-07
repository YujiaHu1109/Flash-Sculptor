from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline

cfg = load_cfg(f'pipe/cfgs/basic.yaml')
# cfg.scene.input.rgb = 'data/sd_readingroom/color.png'
# cfg.scene.input.rgb = 'data/background_recover.png'
cfg.scene.input.rgb = '../results/24/background/bg.png'

cfg.scene.input.rgb = '../results/5/background/background_recover.png'
cfg.scene.input.rgb = '../results/72/2DImage.png'
cfg.scene.input.rgb = '../results/324/2DImage.png'
vistadream = Pipeline(cfg)
vistadream()
