'''
Coarse Gaussian Rendering -- RGB-D as init
RGB-D add noise (MV init)
Cycling:
    denoise to x0 and d0 -- optimize Gaussian
    re-rendering RGB-D
    render RGB-D to rectified noise
    noise rectification
    step denoise with rectified noise
-- Finally the Gaussian
'''
import torch
import numpy as np
from copy import deepcopy
from ops.utils import *
from ops.gs.train import *
from ops.trajs import _generate_trajectory
from ops.gs.basic import Frame,Gaussian_Scene

class Refinement_Tool_MCS():
    def __init__(self,
                 coarse_GS:Gaussian_Scene,
                 device = 'cuda',
                 refiner = None,
                 traj_type = 'spiral',
                 n_view = 8,
                 rect_w = 0.7,
                 n_gsopt_iters = 256) -> None:
        # input coarse GS
        # refine frames to be refined; here we refine frames rather than gaussian paras
        self.n_view = n_view
        self.rect_w = rect_w
        self.n_gsopt_iters = n_gsopt_iters
        self.coarse_GS = coarse_GS
        self.refine_frames: list[Frame] = []
        # hyperparameters total is 50 steps and here is the last N steps
        self.process_res = 512
        self.device = device
        self.traj_type = traj_type
        # models
        self.RGB_LCM = refiner
        self.RGB_LCM.to('cuda')
        self.steps = self.RGB_LCM.denoise_steps
        # prompt for diffusion
        prompt = self.coarse_GS.frames[-1].prompt
        self.rgb_prompt_latent = self.RGB_LCM.model._encode_text_prompt(prompt)
        # loss function
        self.rgb_lossfunc = RGB_Loss(w_ssim=0.2)

    def _pre_process(self): 
        # determine the diffusion target shape
        strict_times = 32
        origin_H = self.coarse_GS.frames[0].H
        origin_W = self.coarse_GS.frames[0].W
        self.target_H,self.target_W = self.process_res,self.process_res
        # reshape to the same (target) shape for rendering and denoising
        intrinsic = deepcopy(self.coarse_GS.frames[0].intrinsic)
        H_ratio, W_ratio = self.target_H/origin_H, self.target_W/origin_W
        intrinsic[0] *= W_ratio
        intrinsic[1] *= H_ratio 
        target_H, target_W = self.target_H+2*strict_times, self.target_W+2*strict_times
        intrinsic[0,-1] = target_W/2
        intrinsic[1,-1] = target_H/2
        # generate a set of cameras
        trajs = _generate_trajectory(None,self.coarse_GS,nframes=self.n_view+2)[1:-1]
        for i, pose in enumerate(trajs):
            fine_frame = Frame()
            fine_frame.H = target_H
            fine_frame.W = target_W
            fine_frame.extrinsic = pose
            fine_frame.intrinsic = deepcopy(intrinsic)
            fine_frame.prompt  = self.coarse_GS.frames[-1].prompt
            self.refine_frames.append(fine_frame) 
        # determine inpaint mask
        temp_scene = Gaussian_Scene()
        temp_scene._add_trainable_frame(self.coarse_GS.frames[0],require_grad=False)
        temp_scene._add_trainable_frame(self.coarse_GS.frames[1],require_grad=False)
        for frame in self.refine_frames:
            frame = temp_scene._render_for_inpaint(frame)
            
    def _mv_init(self):
        rgbs = []
        # only for inpainted images
        for frame in self.refine_frames:
            # rendering at now; all in the same shape
            render_rgb,render_dpt,render_alpha=self.coarse_GS._render_RGBD(frame)
            # diffusion images
            rgbs.append(render_rgb.permute(2,0,1)[None])
        self.rgbs = torch.cat(rgbs,dim=0)
        self.RGB_LCM._encode_mv_init_images(self.rgbs)

    def _to_cuda(self,tensor):
        tensor = torch.from_numpy(tensor.astype(np.float32)).to('cuda')
        return tensor

    def _x0_rectification(self, denoise_rgb, iters):
        # gaussian initialization
        CGS = deepcopy(self.coarse_GS)
        for gf in CGS.gaussian_frames:
            gf._require_grad(True)
        self.refine_GS = GS_Train_Tool(CGS)
        # rectification
        for iter in range(iters):
            loss = 0.
            # supervise on input view
            for i in range(2):
                keep_frame :Frame = self.coarse_GS.frames[i]
                render_rgb,render_dpt,render_alpha = self.refine_GS._render(keep_frame)
                loss_rgb = self.rgb_lossfunc(render_rgb,self._to_cuda(keep_frame.rgb),valid_mask=keep_frame.inpaint)
                loss += loss_rgb*len(self.refine_frames)
            # then multiview supervision
            for i,frame in enumerate(self.refine_frames):
                render_rgb,render_dpt,render_alpha = self.refine_GS._render(frame)
                loss_rgb_item = self.rgb_lossfunc(denoise_rgb[i],render_rgb)
                loss += loss_rgb_item
            # optimization
            loss.backward()  
            self.refine_GS.optimizer.step()
            self.refine_GS.optimizer.zero_grad()
        
    def _step_gaussian_optimization(self,step):
        # denoise to x0 and d0
        with torch.no_grad():
            # we left the last 2 steps for stronger guidances
            rgb_t = self.RGB_LCM.timesteps[-self.steps+step]
            rgb_t = torch.tensor([rgb_t]).to(self.device)
            rgb_noise_pr,rgb_denoise = self.RGB_LCM._denoise_to_x0(rgb_t,self.rgb_prompt_latent)
            rgb_denoise = rgb_denoise.permute(0,2,3,1)
        # rendering each frames and weight-able refinement
        self._x0_rectification(rgb_denoise,self.n_gsopt_iters)      
        return rgb_t, rgb_noise_pr

    def _step_diffusion_rectification(self, rgb_t, rgb_noise_pr):
        # re-rendering RGB
        with torch.no_grad():
            x0_rect = []
            for i,frame in enumerate(self.refine_frames):
                re_render_rgb,_,re_render_alpha= self.refine_GS._render(frame)
                # avoid rasterization holes yield more block holes and more
                x0_rect.append(re_render_rgb.permute(2,0,1)[None])
            x0_rect = torch.cat(x0_rect,dim=0)
        # rectification
        self.RGB_LCM._step_denoise(rgb_t,rgb_noise_pr,x0_rect,rect_w=self.rect_w) 

    def __call__(self):
        # warmup
        self._pre_process()
        self._mv_init()
        for step in tqdm.tqdm(range(self.steps)):
            rgb_t, rgb_noise_pr = self._step_gaussian_optimization(step)
            self._step_diffusion_rectification(rgb_t, rgb_noise_pr)
        scene = self.refine_GS.GS
        for gf in scene.gaussian_frames:
            gf._require_grad(False)
        return scene