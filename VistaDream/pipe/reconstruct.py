'''
Dust3R reconstrucion
GeoWizard Estimation
Smooth Projection
'''
import torch
import PIL,cv2
import numpy as np
from PIL import Image
from ops.gs.basic import Frame
from ops.utils import *
from ops.depth_pro import Depth_Pro_Tool
from ops.connect import Smooth_Connect_Tool


class Reconstruct_Tool():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self._load_model()
        self.connector = Smooth_Connect_Tool()
        
    def _load_model(self):
        self.pro_dpt = Depth_Pro_Tool(ckpt=self.cfg.model.mde.dpt_pro.ckpt,device='cpu')

    def _ProDpt_(self, rgb, intrinsic=None):
        # conduct reconstruction
        print('Pro_dpt[1/3] Move Pro_dpt.model to GPU...')
        self.pro_dpt.to('cuda')
        print('Pro_dpt[2/3] Pro_dpt Estimation...')
        f_px = intrinsic[0,0] if intrinsic is not None else None
        metric_dpt,intrinsic = self.pro_dpt(rgb,f_px)
        print('Pro_dpt[3/3] Move Pro_dpt.model to GPU...')
        self.pro_dpt.to('cpu')
        torch.cuda.empty_cache()
        edge_mask = edge_filter(metric_dpt,times=0.05)
        return metric_dpt, intrinsic, edge_mask

    def _Guide_ProDpt_(self, rgb, intrinsic=None, refer_dpt=None, refer_msk=None):
        # conduct reconstruction
        print('Pro_dpt[1/3] Move Pro_dpt.model to GPU...')
        self.pro_dpt.to('cuda')
        print('Pro_dpt[2/3] Pro_dpt Estimation...')
        f_px = intrinsic[0,0] if intrinsic is not None else None
        metric_dpt,intrinsic = self.pro_dpt(rgb,f_px=f_px)
        metric_dpt_connect = self.connector._affine_dpt_to_GS(refer_dpt,metric_dpt,~refer_msk)
        print('Pro_dpt[3/3] Move Pro_dpt.model to GPU...')
        self.pro_dpt.to('cpu')
        torch.cuda.empty_cache()
        edge_mask = edge_filter(metric_dpt_connect,times=0.05)
        return metric_dpt_connect, metric_dpt, intrinsic, edge_mask
    
    # ------------- TODO: Metricv2 + Guide-GeoWizard ------------------ #
