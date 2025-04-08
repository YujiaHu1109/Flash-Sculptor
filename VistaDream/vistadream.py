import argparse
import torch
import os
from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline
from ops.visual_check import Check
from ops.utils import save_ply
import argparse
import numpy as np
from plyfile import PlyData
    
def extract_points_by_ids(points, gaussian_ids):
    vertices = points['vertex']
    selected_points = vertices[gaussian_ids]
    return selected_points

def read_ply(filename):
    return PlyData.read(filename)

def main():
    parser = argparse.ArgumentParser(description="3D bg reconstruction.")
    parser.add_argument("--task_name", type=str, required=True, help="Task name.")
    args = parser.parse_args()

    cfg = load_cfg(f'pipe/cfgs/basic.yaml')
    cfg.scene.input.rgb = f'../results/{args.task_name}/background/background_recover.png'

    vistadream = Pipeline(cfg)
    vistadream()

    # json_path = os.path.join("results", args.task_name, "SAM", "label.json")
    scene = torch.load(f'../results/{args.task_name}/background/scene.pth')
    check = Check()
    gaussian_ids = check._render_video(scene, save_dir="./")

    visible_path = f'../results/{args.task_name}/mask.npy'
    np.save(visible_path, gaussian_ids)

    ply_path = f'../results/{args.task_name}/background/point_cloud.ply'
    save_ply(scene, ply_path)

    scene_ply = read_ply(ply_path)
    visible_points = extract_points_by_ids(scene_ply, gaussian_ids)
    x_values = visible_points['x']
    y_values = visible_points['y']
    z_values = visible_points['z']

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)
    z_min, z_max = np.min(z_values), np.max(z_values)

    x_range = x_max - x_min 
    y_range = y_max - y_min
    z_range = z_max - z_min

    print("Visible points:")
    print(f"x: min={x_min}, max={x_max}, range={x_range}")
    print(f"y: min={y_min}, max={y_max}, range={y_range}")
    print(f"z: min={z_min}, max={z_max}, range={z_range}")

    # z range
    folder_path = f"../results/{args.task_name}/SAM/"
    npy_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('mask_') and f.endswith('.npy')]

    if npy_files:
        mask_array = np.load(npy_files[0])
        for file in npy_files[1:]:
            current_mask = np.load(file)
            mask_array = np.logical_or(mask_array, current_mask)
        mask_array = mask_array.astype(np.uint8)
    mask_array = mask_array.squeeze(0)

    depth_array = np.load(f"../results/{args.task_name}/depth/2DImage_pred.npy")
    valid_depth = depth_array[mask_array == 1]
    mean_depth = np.mean(valid_depth)
    std_depth = np.std(valid_depth)

    z_scores = np.abs((valid_depth - mean_depth) / std_depth)

    threshold = 3

    filtered_depth = valid_depth[z_scores < threshold]
    range_z = np.max(filtered_depth) - np.min(filtered_depth)
    axis_scale_z = range_z * z_range / x_range
    axis_scale_z = np.array([axis_scale_z])
    np.save(f"../results/{args.task_name}/axis_scale.npy", axis_scale_z)

    scene = torch.load(ply_path)
  
    print(dir(scene))
    print(scene.frames)

    check = Check()
    visible_points = check._render_visible(scene, save_dir="./")
    np.save(f"../results/{args.task_name}/visible_points.npy", visible_points)
    
    


