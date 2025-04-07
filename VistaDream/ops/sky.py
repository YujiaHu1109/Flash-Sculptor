import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/OneFormer'
sys.path.insert(0,reference)

from oneformer_command import OneFormer_Segment

class Sky_Seg_Tool():
    def __init__(self,cfg):
        self.ckpt = cfg.model.sky.oneformer.ckpt
        self.yaml = cfg.model.sky.oneformer.yaml
        # ckpt='/mnt/proj/GaussianAnythingV2/Tools/OneFormer/ckpts/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth',
        # yaml='/mnt/proj/GaussianAnythingV2/Tools/OneFormer/configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml') -> None:
        self.segor = OneFormer_Segment(self.ckpt,self.yaml)
    
    def __call__(self, rgb):
        '''
        input rgb should be numpy in range of 0-1 or 0-255
        '''
        return self.segor(rgb)
        