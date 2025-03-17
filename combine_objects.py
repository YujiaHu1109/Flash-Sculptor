# for 3DGS
import os
import re
import json
import trimesh
import argparse
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
# import seaborn as sns

target_dtype = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('nx', np.float32),
    ('ny', np.float32),
    ('nz', np.float32),
    ('f_dc_0', np.float32),
    ('f_dc_1', np.float32),
    ('f_dc_2', np.float32),
    ('opacity', np.float32),
    ('scale_0', np.float32),
    ('scale_1', np.float32),
    ('scale_2', np.float32),
    ('rot_0', np.float32),
    ('rot_1', np.float32),
    ('rot_2', np.float32),
    ('rot_3', np.float32),
])

def load_and_transform_ply(ply_dir, file_path, scale_factor, translation_vector, rotation_angles=(0, 0, 0)):
    ply_data = PlyData.read(os.path.join(ply_dir, file_path))
    vertices = ply_data['vertex'].data
    print(vertices.shape)
    new_vertices = np.zeros(vertices.shape[0], dtype=target_dtype)
    print(file_path, scale_factor)

    for field in target_dtype.names:
        if field in vertices.dtype.names:
            # scale
            if field in ['x', 'y', 'z']:
                new_vertices[field] = vertices[field] * scale_factor
            elif field in ['scale_0', 'scale_1', 'scale_2']:
                new_vertices[field] = vertices[field] + np.log(scale_factor) 
            else:
                new_vertices[field] = vertices[field]
            if field == 'x':
                new_vertices[field] = new_vertices[field] + translation_vector[0]
            elif field == 'y':
                new_vertices[field] = new_vertices[field] + translation_vector[1]
            elif field == 'z':
                new_vertices[field] = new_vertices[field] + translation_vector[2]

    return new_vertices  

def combine_meshes_with_properties(mesh_list):
    total = 0
    for i in range (len(mesh_list)):
        total += mesh_list[i].shape[0]
    
    # print(total)

    new_vertices = np.zeros(total, dtype=target_dtype)
    current_index = 0
    for i in range(len(mesh_list)):
        vertexs = mesh_list[i]
        for field in target_dtype.names:
            if field in vertexs.dtype.names:
                new_vertices[field][current_index:current_index + len(vertexs)] = vertexs[field]

        current_index += len(vertexs)
    return new_vertices

def save_ply(filename, vertices):
    new_element = PlyElement.describe(vertices, 'vertex')
    PlyData([new_element], text=False).write(filename)

def load_ply(ply_dir, ply_files, index):
    pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_files[index]))

    points = np.asarray(pcd.points)

    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)

    return min_values, max_values

def calculate_visible(ply_dir, ply_files, index):
    pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_files[index]))

    viewpoint = np.array([0, 0, 1]) 

    radius = 50  
    _, pt_map = pcd.hidden_point_removal(viewpoint, radius)

    sorted_pt_map = np.sort(pt_map)
    front_surface_pcd = pcd.select_by_index(sorted_pt_map)
    print(len(front_surface_pcd.points))

    points = np.asarray(front_surface_pcd.points)
    indices = np.array(sorted_pt_map)

    return points, indices

def normalize_xy(z, a, b):
    normalized_value = a + (b - a) * z
    return normalized_value

def find_closest_point(visible_points, current_x, current_y):
    distances = np.linalg.norm(visible_points[:, :2] - np.array([current_x, current_y]), axis=1)
    closest_index = np.argmin(distances)
    # print(np.min(distances))
    
    return visible_points[closest_index, 2], closest_index, distances[closest_index]

def remove_outliers_mad(z_moves, threshold=3.0):
    if not isinstance(z_moves, np.ndarray):
        z_moves = np.array(z_moves)
    median = np.median(z_moves)
    abs_deviations = np.abs(z_moves - median)
    mad = np.median(abs_deviations)
    mad_normalized = 1.4826 * mad
    z_scores = abs_deviations / mad_normalized
    print("z_scores shape:", z_scores.shape)
    print("z_moves shape:", z_moves.shape)
    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a numeric type (int or float)")
    inlier_indices = np.where(z_scores < threshold)[0]
    outlier_indices = np.where(z_scores >= threshold)[0]
    non_outliers = z_moves[inlier_indices]
    return non_outliers, outlier_indices

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--task_name", type=str, required=True, help="Task name.")

    args = parser.parse_args()
    ply_dir = os.path.join("results", args.task_name, "Single3DN")
    json_path = os.path.join("results", args.task_name, "SAM", "label.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
    print(ply_files)
    num_objects = len(ply_files)
    
    # rotation angels
    rotation_angles = np.zeros((num_objects, 3))
    # rotation_angles = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]  
    # print(rotation_angles.shape)

    # scaling factors
    numbers = []
    for file in ply_files:
        match = re.search(r'\d+', file)
        if match:
            numbers.append(int(match.group()))

    print(f"numbers: {numbers}")
    scale_pre = []
  
    x_min_out = float('inf')
    y_min_out = float('inf')
    x_max_out = float('-inf')
    y_max_out = float('-inf')
    for i in range(num_objects):
        box = [item['box'] for item in data['mask'] if item['value'] == numbers[i]+1]
        xx = box[0][2] - box[0][0]
        yy = box[0][3] - box[0][1]
        x_min_out = min(x_min_out, box[0][0])  
        y_min_out = min(y_min_out, box[0][1]) 
        x_max_out = max(x_max_out, box[0][2]) 
        y_max_out = max(y_max_out, box[0][3])
        print(i, xx, yy)
        xym = max(xx, yy)
        scale_pre.append(xym)

    base_scale = max(scale_pre)
    index_max = scale_pre.index(base_scale)

    mesh_scale = []
    for i in range(num_objects):
        min_values, max_values = load_ply(ply_dir, ply_files, i) # numbers[i]
        mesh_scale.append(max((max_values[0]-min_values[0]), (max_values[1]-min_values[1])))

    # print("mesh scale")
    # print(mesh_scale)

    scale_factors = [value / base_scale for value in scale_pre]
    # print("scale factors:")
    # print(scale_factors)
    scale_factors_np = np.array(scale_factors)
    mesh_scale_np = np.array(mesh_scale)
    scale_factors = scale_factors_np / mesh_scale_np

    # calculate depth
    depth_average = []
    for i in range(num_objects):
        num = numbers[i]
        scale_factor = scale_factors[i]
        print(num, scale_factor)
        points = np.load(os.path.join("results", args.task_name, "Single3DN", f"points_cord_{num}.npy"))
        points_z = points[:, 2]
        depth_average.append(np.mean(points_z))
    # print(depth_average)

    
    depth_average = np.array(depth_average)
    depth_min = np.min(depth_average)
    depth_average_scale = depth_average/depth_min
    # print(depth_average_scale)

    scale_factors_nd = scale_factors
    # print(f"scale factor: {scale_factors}")
    # scale_factors = scale_factors * depth_average_scale

    scale_factors = scale_factors.tolist()
    print(f"scale factor: {scale_factors}") # final

    min_values, max_values = load_ply(ply_dir, ply_files, index_max)
    print((max_values[0]-min_values[0]), (max_values[1]-min_values[1]))
    mesh_scale_max = max((max_values[0]-min_values[0]), (max_values[1]-min_values[1]))
    print(mesh_scale_max * scale_factors_nd[index_max])
    print(scale_factors[index_max])

    max_axis_scale = (mesh_scale_max * scale_factors[index_max]/(scale_pre[index_max]/1024))
    print(f"max_axis_scale: {max_axis_scale}")
    max_axis_scale = max_axis_scale 

    translations = np.zeros((num_objects, 3)) 
    translation_path = os.path.join("results", args.task_name, "SAM", "translation_2DImage.npy")
    translations_ori = np.load(translation_path)
    print(translations_ori)

    for i in range(num_objects):
        num = numbers[i]
        scale_factor = scale_factors[i] 

        x_target = translations_ori[num][0]
        x_target = x_target - 0.5
        y_target = translations_ori[num][1]
        y_target = 0.5 - y_target
        z_target = 1-translations_ori[num][2] 
        z_target = 0.5 - z_target
        x_target = x_target * max_axis_scale 
        y_target = y_target * max_axis_scale 
        z_target = z_target * max_axis_scale

        min_values, max_values = load_ply(ply_dir, ply_files, i)

        x_ori = (min_values[0] + max_values[0])/2
        y_ori = (min_values[1] + max_values[1])/2
        # print(x_ori, y_ori)
        x_move = x_target - x_ori * scale_factor
        y_move = y_target - y_ori * scale_factor
        translations[i][0] = x_move
        translations[i][1] = y_move

    # adjust z
    axis_scale_z_arr = np.load(f"./results/{args.task_name}/axis_scale.npy")
    axis_scale_z = axis_scale_z_arr[0]
    z_oris = []
    z_targets = []
    for i in range(num_objects):
        num = numbers[i]
        scale_factor = scale_factors[i]
        print(num, scale_factor)
        points = np.load(os.path.join("results", args.task_name, "Single3DN", f"points_cord_{num}.npy"))
        points_xy = points[:, :2]     
        
        min_v, max_v = load_ply(ply_dir, ply_files, num) 
 
        points_visible, visible_indices = calculate_visible(ply_dir, ply_files, i)

        z_moves = []
        indexs = []
        for j in range(500):
            x_nor = normalize_xy(points_xy[j][0], min_v[0], max_v[0])
            y_nor = normalize_xy(1-points_xy[j][1], min_v[1], max_v[1])
            
            z_ori, index, dis = find_closest_point(points_visible, x_nor, y_nor)
            z_oris.append(z_ori)
            if dis < 1:
                indexs.append(index)
                # z_target = 1-points[j][2] 
                z_target = points[j][2]
                z_targets.append(z_target)
                z_target = (0.5 - z_target) * max_axis_scale * axis_scale_z
                z_move = z_target - z_ori * scale_factor
                z_moves.append(z_move)
            
        print(np.mean(z_moves), np.std(z_moves))

        filtered_z_moves, outlier_indices = remove_outliers_mad(np.array(z_moves))
        final_z_move = np.mean(filtered_z_moves)
        translations[i][2] = final_z_move
        # print(final_z_move)
    z_oris = np.array(z_oris)
    z_targets = np.array(z_targets)
    np.save(f"results/{args.task_name}/z_oris.npy", z_oris)
    np.save(f"results/{args.task_name}/z_targets.npy", z_targets)

    meshes = []
    for i, file_path in enumerate(ply_files):
        mesh = load_and_transform_ply(
                ply_dir,
                file_path, 
                scale_factor=scale_factors[i], 
                translation_vector=translations[i], 
                rotation_angles=rotation_angles[i]
        )

        meshes.append(mesh)

    combined_mesh = combine_meshes_with_properties(meshes)
    
    save_ply(f"./results/{args.task_name}/output.ply", combined_mesh)

if __name__ == "__main__":
    main()

