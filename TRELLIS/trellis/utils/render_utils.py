import torch
import numpy as np
from tqdm import tqdm
import utils3d
import trimesh
import imageio
from PIL import Image

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence, get_icosphere_spherical_coords


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
        else:
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
    return rets

def render_single(sample, resolution=256):
    r = 2  
    fov = 40  

    phi_values = [0]  
    theta_values = [0] 

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(phi_values, theta_values, r, fov)

    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})

    return res['color'], extrinsics, intrinsics


def render_video(sample, resolution=512, bg_color=(1, 1, 1), num_frames=300, r=3, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})
    return res['color'], extrinsics, intrinsics

def render_multiview_210(sample, resolution=512):
    Radius = 2.2
    fovs = [60, 45, 36, 30, 24]
    cams = get_icosphere_spherical_coords()
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics = []
    intrinsics = []
    for i in range(5):
        ex, int = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, Radius, fovs[i])
        extrinsics += ex
        intrinsics += int
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})

    for idx in range(len(res['color'])):
        group_idx = idx // 5 
        sub_idx = idx % 5
        color = Image.fromarray(res['color'][idx].astype(np.uint8))
        imageio.imwrite(f'output/{group_idx:03d}_{sub_idx}.png', color)

    return res['color']

def render_multiview_12(sample, resolution=512):
    Radius = 2.2
    fovs = [60]
    cams = get_icosphere_spherical_coords(sub=0)
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics = []
    intrinsics = []
    for i in range(1):
        ex, int = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, Radius, fovs[i])
        extrinsics += ex
        intrinsics += int
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})

    for idx, color in enumerate(res['color']):
        color = Image.fromarray(color.astype(np.uint8))
        imageio.imwrite(f'outputcap/{idx:03d}.png', color)

    return res['color']

def render_4view(sample, resolution=512):
    r = 2  
    fov = 40  

    phi_values = [0, np.pi / 2, -np.pi / 2, np.pi]  
    theta_values = [0, 0, 0, 0]  

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(phi_values, theta_values, r, fov)

    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})

    return res['color'], extrinsics, intrinsics

def render_multi_view(sample, resolution=1024):
    r = 2  
    fov = 40 
    yaw_angles = np.arange(0, 360, 30) * np.pi / 180  
    pitch_angles = np.arange(-90, 60, 30) * np.pi / 180  

    phi_values, theta_values = np.meshgrid(yaw_angles, pitch_angles, indexing='ij')
    phi_values, theta_values = phi_values.ravel(), theta_values.ravel()
    phi_values = phi_values.tolist()
    theta_values = theta_values.tolist()

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(phi_values, theta_values, r, fov)

    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})

    return res['color'], extrinsics, intrinsics

def render_5view(sample, resolution=1024):
    r = 2  # 相机距离
    fov = 40  # 视场角

    # 定义正面、两个侧面和背面的视角（phi 和 theta）
    phi_values = [0, np.pi / 4, np.pi / 2, -np.pi / 2, np.pi]  # 正面、右侧、左侧、背面
    theta_values = [0, 0, 0, 0, 0]  # 俯仰角均为 0（水平视角）

    # 计算相机外参和内参
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(phi_values, theta_values, r, fov)

    # 渲染图像
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})

    return res['color'], extrinsics, intrinsics

def render_6view(sample, resolution=1024):
    r = 2  
    fov = 40  

    phi_values = [0, np.pi / 4, np.pi / 2, np.pi *0.75,-np.pi / 2, -np.pi/4]  
    theta_values = [-30, -30, -30, -30, -30, -30]  

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(phi_values, theta_values, r, fov)
    print(len(extrinsics), len(intrinsics))
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (1, 1, 1)})

    return res['color'], extrinsics, intrinsics

def rotate_model(gaussian_obj, axis, angle):
    vertices = gaussian_obj.get_xyz  # [N, 3] 
    center = vertices.mean(dim=0, keepdim=True)  # [1, 3]
    vertices_centered = vertices - center

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    if axis == 'x':
        rotation_matrix = torch.tensor([[1, 0, 0],
                                       [0, cos_a, -sin_a],
                                       [0, sin_a, cos_a]], dtype=torch.float32)
    elif axis == 'y':
        rotation_matrix = torch.tensor([[cos_a, 0, sin_a],
                                       [0, 1, 0],
                                       [-sin_a, 0, cos_a]], dtype=torch.float32)
    elif axis == 'z':
        rotation_matrix = torch.tensor([[cos_a, -sin_a, 0],
                                       [sin_a, cos_a, 0],
                                       [0, 0, 1]], dtype=torch.float32)
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    device = vertices.device
    rotation_matrix = rotation_matrix.to(device)

    rotated_vertices = torch.matmul(vertices_centered, rotation_matrix.T)  # [N, 3]

    rotated_vertices = rotated_vertices + center

    gaussian_obj._xyz = rotated_vertices.to(device)

    return gaussian_obj

def render_sparse(sample, resolution=256):
    r = 2
    fov = 40
    x_angles = np.array([0])
    images = []

    for x_angle in x_angles:
        yaw = 0
        pitch = -80
        extrinsics = []
        intrinsics = []
    
        rotated_sample = rotate_model(sample, 'x', np.radians(x_angle))  # 仅绕x轴旋转

        ext, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)

        extrinsics.append(ext)
        intrinsics.append(intr)
   
        rendered_frame = render_frames(rotated_sample, torch.stack(extrinsics), torch.stack(intrinsics), 
                                       {'resolution': resolution, 'bg_color': (1, 1, 1)})
        # images.append(rendered_frame['color'])
        print(len(rendered_frame['color']))
        images.append(rendered_frame['color'][0])

    return images, torch.stack(extrinsics), torch.stack(intrinsics)

def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)
