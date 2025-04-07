import cv2
import numpy as np
from copy import deepcopy
from ops.utils import dpt2xyz,transform_points

class Connect_Tool():
    def __init__(self) -> None:
        pass
        
    def _align_scale_shift_numpy(self, pred: np.array, target: np.array):
        mask = (target > 0) & (pred < 199)
        target_mask = target[mask]
        pred_mask = pred[mask]
        if np.sum(mask) > 10:
            scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
            if scale < 0:
                scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
                shift = 0
        else:
            scale = 1
            shift = 0
        return scale,shift
        
    def __call__(self, render_dpt, inpaint_dpt, inpaint_msk):
        if np.sum(inpaint_msk > 0.5) < 1.: return render_dpt
        # get areas need to be aligned
        render_dpt_valid  = render_dpt[~inpaint_msk]
        inpaint_dpt_valid = inpaint_dpt[~inpaint_msk]
        # rectify
        scale,shift = self._align_scale_shift_numpy(inpaint_dpt_valid,render_dpt_valid)
        inpaint_dpt = inpaint_dpt*scale + shift
        return inpaint_dpt

class Smooth_Connect_Tool():
    def __init__(self) -> None:
        self.coarse_align = Connect_Tool()
    
    def _coarse_alignment(self, render_dpt, ipaint_dpt, ipaint_msk):
        # determine the scale and shift of inpaint_dpt to coarsely align it to render_dpt
        inpaint_dpt = self.coarse_align(render_dpt,ipaint_dpt,ipaint_msk)
        return inpaint_dpt
    
    def _refine_movements(self, render_dpt, ipaint_dpt, ipaint_msk):
        '''
        Follow https://arxiv.org/pdf/2311.13384
        '''
        # Determine the adjustment of un-inpainted area
        ipaint_msk = ipaint_msk>.5
        H, W = ipaint_msk.shape[0:2]
        U = np.arange(W)[None,:].repeat(H,axis=0)
        V = np.arange(H)[:,None].repeat(W,axis=1)
        # on kept areas
        keep_render_dpt = render_dpt[~ipaint_msk]
        keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]
        keep_adjust_dpt = keep_render_dpt - keep_ipaint_dpt
        # iterative refinement
        complete_adjust = np.zeros_like(ipaint_dpt)
        for i in range(100):
            complete_adjust[~ipaint_msk] = keep_adjust_dpt
            complete_adjust = cv2.blur(complete_adjust,(15,15))
        # complete_adjust[~ipaint_msk] = keep_adjust_dpt
        ipaint_dpt = ipaint_dpt + complete_adjust        
        return ipaint_dpt
           
    def _affine_dpt_to_GS(self, render_dpt, inpaint_dpt, inpaint_msk):
        if np.sum(inpaint_msk > 0.5) < 1.: return render_dpt
        inpaint_dpt = self._coarse_alignment(render_dpt,inpaint_dpt,inpaint_msk)
        inpaint_dpt = self._refine_movements(render_dpt,inpaint_dpt,inpaint_msk)
        return inpaint_dpt
                     
    def _scale_dpt_to_GS(self, render_dpt, inpaint_dpt, inpaint_msk):
        if np.sum(inpaint_msk > 0.5) < 1.: return render_dpt
        inpaint_dpt = self._refine_movements(render_dpt,inpaint_dpt,inpaint_msk)
        return inpaint_dpt
    
class Occlusion_Removal():
    def __init__(self) -> None:
        pass
    
    def __call__(self,scene,frame):
        # first get xyz of the newly added frame
        xyz = dpt2xyz(frame.dpt,frame.intrinsic)
        # we only check newly added areas
        xyz = xyz[frame.inpaint]
        # move these xyzs to world coor system
        inv_extrinsic = np.linalg.inv(frame.extrinsic)
        xyz = transform_points(xyz,inv_extrinsic)
        # we will add which pixels to the gaussian scene
        msk = np.ones_like(xyz[...,0])
        # project the xyzs to already built frames
        for former_frame in scene.frames:
            # xyz in camera frustrum
            xyz_camera = transform_points(deepcopy(xyz),former_frame.extrinsic)
            # uvz in camera frustrum
            uvz_camera = np.einsum(f'ab,pb->pa',former_frame.intrinsic,xyz_camera)
            # uv and d in camra frustrum
            uv,d = uvz_camera[...,:2]/uvz_camera[...,-1:], uvz_camera[...,-1]
            # in-frusturm pixels
            valid_msk = (uv[...,0]>0) & (uv[...,0]<former_frame.W) & (uv[...,1]>0) & (uv[...,1]<former_frame.H) & (d>1e-2)
            valid_idx = np.where(valid_msk)[0]
            uv,d = uv[valid_idx].astype(np.uint32),d[valid_idx]            
            # make comparsion: compare_d < d is ok -- compare_d - d < 0(or a small number)    
            compare_d = former_frame.dpt[uv[:,1],uv[:,0]]
            remove_msk = (compare_d-d)>(d+compare_d)/2./15.
            # else to unvalid pixels
            invalid_idx = valid_idx[remove_msk]
            msk[invalid_idx] = 0.
        # USE indexes rather than [][]
        inpaint_idx_v,inpaint_idx_u = np.where(frame.inpaint)
        inpaint_idx_v = inpaint_idx_v[msk<.5]
        inpaint_idx_u = inpaint_idx_u[msk<.5]
        frame.inpaint[inpaint_idx_v,inpaint_idx_u] = False
        return frame