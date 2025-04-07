import imageio
import matplotlib
from ops.utils import *
from ops.gs.basic import *
from ops.trajs import _generate_trajectory
from pathlib import Path

class Check():
    def __init__(self) -> None:
        pass
        
    def _visual_pcd(self,scene:Gaussian_Scene):
        xyzs,rgbs = [],[]
        for i,gf in enumerate(scene.gaussian_frames):
            xyz = gf.xyz.detach().cpu().numpy()
            rgb = torch.sigmoid(gf.rgb).detach().cpu().numpy()
            opacity = gf.opacity.detach().squeeze().cpu().numpy() > 1e-5
            xyzs.append(xyz[opacity])
            rgbs.append(rgb[opacity])
        xyzs = np.concatenate(xyzs,axis=0)
        rgbs = np.concatenate(rgbs,axis=0)
        visual_pcd(xyzs,color=rgbs,normal=True)
        
    @torch.no_grad()
    def _render_video(self,scene:Gaussian_Scene,save_dir='./'):
        # render 5times frames
        nframes = len(scene.frames)*25
        video_trajs = _generate_trajectory(None,scene,nframes=nframes)
        H,W,intrinsic = scene.frames[0].H,scene.frames[0].W,deepcopy(scene.frames[0].intrinsic)
        if H<W:
            if H>512:
                ratio = 512/H
                W,H = int(W*ratio),int(H*ratio)
                intrinsic[0:2] = intrinsic[0:2]*ratio
        else:
            if W>512:
                ratio = 512/W
                W,H = int(W*ratio),int(H*ratio)
                intrinsic[0:2] = intrinsic[0:2]*ratio
        # render
        rgbs,dpts = [],[]
        print(f'[INFO] rendering final video with {nframes} frames...')
        first_iteration = True
        for pose in video_trajs:
            # print(pose)
            # print(intrinsic)
            # exit(0)
            frame = Frame(H=H,W=W,
                          intrinsic=intrinsic,
                          extrinsic=pose)
            rgb,dpt,alpha,gaussian_ids= scene._render_RGBD(frame)
            # print(type(gaussian_ids))
            # print(gaussian_ids.shape)
            # visible_points = gaussian_ids.cpu().numpy()
            # print(visible_points.shape)
            if first_iteration:
                gau = gaussian_ids.cpu().numpy()
                first_iteration = False
            rgb = rgb.detach().float().cpu().numpy()
            dpt = dpt.detach().float().cpu().numpy()
            dpts.append(dpt)
            rgbs.append((rgb * 255).astype(np.uint8))
        rgbs = np.stack(rgbs, axis=0)
        dpts = np.stack(dpts, axis=0)
        valid_dpts = dpts[dpts>0.]
        _min = np.percentile(valid_dpts, 1)
        _max = np.percentile(valid_dpts,99)
        dpts = (dpts-_min) / (_max-_min)
        dpts = dpts.clip(0,1)

        cm = matplotlib.colormaps["plasma"]
        dpts_color = cm(dpts,bytes=False)[...,0:3]
        dpts_color = (dpts_color*255).astype(np.uint8)
            
        imageio.mimwrite(f'{save_dir}video_rgb.mp4',rgbs,fps=20)
        imageio.mimwrite(f'{save_dir}video_dpt.mp4',dpts_color,fps=20)
        return gau
    
    @torch.no_grad()
    def _render_images(self, scene: Gaussian_Scene, save_dir='./', colorize=True):

        # 创建保存文件夹
        rendered_dir = Path(save_dir) / "rendered"
        rendered_dir.mkdir(parents=True, exist_ok=True)

        # render 5 times frames
        nframes = len(scene.frames) * 25
        video_trajs = _generate_trajectory(None, scene, nframes=nframes)
        H, W, intrinsic = scene.frames[0].H, scene.frames[0].W, deepcopy(scene.frames[0].intrinsic)
        if H < W:
            if H > 512:
                ratio = 512 / H
                W, H = int(W * ratio), int(H * ratio)
                intrinsic[0:2] = intrinsic[0:2] * ratio
        else:
            if W > 512:
                ratio = 512 / W
                W, H = int(W * ratio), int(H * ratio)
                intrinsic[0:2] = intrinsic[0:2] * ratio

        # render
        # intrinsic[0][0] = 1500
        # intrinsic[1][1] = 1500
        print(intrinsic)
        # exit(0)
        print(video_trajs[0])
        # exit(0)
        print(f'[INFO] rendering final {nframes} frames...')
        ex_matrix = np.load("transforms.npy")
        video_trajs = [ex_matrix[i] for i in range(ex_matrix.shape[0])]
        for i, pose in enumerate(video_trajs):
            frame = Frame(H=H, W=W, intrinsic=intrinsic, extrinsic=np.linalg.inv(pose))
            rgb, dpt, alpha = scene._render_RGBD(frame)
            rgb = rgb.detach().float().cpu().numpy()
            dpt = dpt.detach().float().cpu().numpy()
            
            # 保存RGB图像
            rgb_image = (rgb * 255).astype(np.uint8)
            imageio.imwrite(rendered_dir / f"frame_{i:04d}_rgb.png", rgb_image)
            
            # 处理深度图
            valid_dpts = dpt[dpt > 0.]
            _min = np.percentile(valid_dpts, 1)
            _max = np.percentile(valid_dpts, 99)
            dpt = (dpt - _min) / (_max - _min)
            dpt = dpt.clip(0, 1)

            if colorize:
                cm = matplotlib.colormaps["viridis"]
                dpts_color = cm(dpt, bytes=False)[..., 0:3]
                dpt_image = (dpts_color * 255).astype(np.uint8)
            else:
                dpt_image = (dpt[..., None].repeat(3, axis=-1) * 255).astype(np.uint8)
            
            # 保存深度图
            imageio.imwrite(rendered_dir / f"dep_frame_{i:04d}_dpt.png", dpt_image)

        print(f'[INFO] Rendered frames saved in {rendered_dir}')
    
    @torch.no_grad()
    def _render_visible(self,scene:Gaussian_Scene,save_dir='./'):
        # render 5times frames
        nframes = len(scene.frames)*25
        video_trajs = _generate_trajectory(None,scene,nframes=nframes)
        H,W,intrinsic = scene.frames[0].H,scene.frames[0].W,deepcopy(scene.frames[0].intrinsic)
        if H<W:
            if H>512:
                ratio = 512/H
                W,H = int(W*ratio),int(H*ratio)
                intrinsic[0:2] = intrinsic[0:2]*ratio
        else:
            if W>512:
                ratio = 512/W
                W,H = int(W*ratio),int(H*ratio)
                intrinsic[0:2] = intrinsic[0:2]*ratio
        # render
        rgbs,dpts = [],[]
        print(f'[INFO] rendering final video with {nframes} frames...')
        first_iteration = True
        for pose in video_trajs:
            # print(pose)
            # print(intrinsic)
            # exit(0)
            frame = Frame(H=H,W=W,
                          intrinsic=intrinsic,
                          extrinsic=pose)
            rgb,dpt,alpha,gaussian_ids= scene._render_RGBD(frame)
            # print(type(gaussian_ids))
            # print(gaussian_ids.shape)
            visible_points = gaussian_ids.cpu().numpy()
            # print(visible_points.shape)
            return visible_points