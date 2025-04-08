import numpy as np
import os
import json
from plyfile import PlyData, PlyElement
import cv2
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import grey_dilation
from PIL import Image
import argparse
import random

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
    ('f_rest_0', np.float32),
    ('f_rest_1', np.float32),
    ('f_rest_2', np.float32),
    ('f_rest_3', np.float32),
    ('f_rest_4', np.float32),
    ('f_rest_5', np.float32),
    ('f_rest_6', np.float32),
    ('f_rest_7', np.float32),
    ('f_rest_8', np.float32),
    ('f_rest_9', np.float32),
    ('f_rest_10', np.float32),
    ('f_rest_11', np.float32),
    ('f_rest_12', np.float32),
    ('f_rest_13', np.float32),
    ('f_rest_14', np.float32),
    ('f_rest_15', np.float32),
    ('f_rest_16', np.float32),
    ('f_rest_17', np.float32),
    ('f_rest_18', np.float32),
    ('f_rest_19', np.float32),
    ('f_rest_20', np.float32),
    ('f_rest_21', np.float32),
    ('f_rest_22', np.float32),
    ('f_rest_23', np.float32),
    ('f_rest_24', np.float32),
    ('f_rest_25', np.float32),
    ('f_rest_26', np.float32),
    ('f_rest_27', np.float32),
    ('f_rest_28', np.float32),
    ('f_rest_29', np.float32),
    ('f_rest_30', np.float32),
    ('f_rest_31', np.float32),
    ('f_rest_32', np.float32),
    ('f_rest_33', np.float32),
    ('f_rest_34', np.float32),
    ('f_rest_35', np.float32),
    ('f_rest_36', np.float32),
    ('f_rest_37', np.float32),
    ('f_rest_38', np.float32),
    ('f_rest_39', np.float32),
    ('f_rest_40', np.float32),
    ('f_rest_41', np.float32),
    ('f_rest_42', np.float32),
    ('f_rest_43', np.float32),
    ('f_rest_44', np.float32),
    ('opacity', np.float32),
    ('scale_0', np.float32),
    ('scale_1', np.float32),
    ('scale_2', np.float32),
    ('rot_0', np.float32),
    ('rot_1', np.float32),
    ('rot_2', np.float32),
    ('rot_3', np.float32),
])

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

def read_ply(filename):
    return PlyData.read(filename)

def rotate_points_180_x(plydata):
    vertices = plydata['vertex'].data

    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    points = np.vstack((x, y, z)).T

    rotated_points = np.dot(points, rotation_matrix.T)

    vertices['x'] = rotated_points[:, 0]
    vertices['y'] = rotated_points[:, 1]
    vertices['z'] = rotated_points[:, 2]

    new_vertices = np.array([tuple(v) for v in vertices], dtype=vertices.dtype)
    vertex_element = PlyElement.describe(new_vertices, 'vertex')

    new_plydata = PlyData([vertex_element], text=plydata.text)

    return new_plydata

def read_ply_and_normalize(filename):
    ply = PlyData.read(filename)
    
    vertex = ply['vertex']
    
    data = np.array(vertex.data)

    x, y, z = data['x'], data['y'], data['z']
    center_x, center_y, center_z = (np.max(x)+np.min(x))/2, (np.max(y)+np.min(y))/2, (np.max(z)+np.min(z))/2

    data['x'] -= center_x
    data['y'] -= center_y
    data['z'] -= center_z

    new_ply = PlyData([PlyElement.describe(data, 'vertex')], text=True)

    return new_ply

def normalize_column(column, new_min, new_max):
    col_min = np.min(column)
    col_max = np.max(column)
    normalized = (column - col_min) / (col_max - col_min)

    return normalized * (new_max - new_min) + new_min

def save_ply(filename, vertices):
    new_element = PlyElement.describe(vertices, 'vertex')
    PlyData([new_element], text=False).write(filename)

def extract_coordinates(ply_data):
    vertices = ply_data['vertex'].data
    coordinates = np.array([(vertex['x'], vertex['y'], vertex['z']) for vertex in vertices], dtype=np.float32)
    return coordinates

def extract_points_by_ids(points, gaussian_ids):
    vertices = points['vertex']
    selected_points = vertices[gaussian_ids]
    return selected_points


def get_bounding_box(bboxes):
    min_x0, min_y0, max_x1, max_y1 = bboxes[0]
        
    for bbox in bboxes[1:]:
        x0, y0, x1, y1 = bbox
        min_x0 = min(min_x0, x0)
        min_y0 = min(min_y0, y0)
        max_x1 = max(max_x1, x1)
        max_y1 = max(max_y1, y1)
        
    return min_x0, min_y0, max_x1, max_y1

def save_points_to_ply_plyfile(points, output_path):
    vertices = np.array(
        list(zip(points['x'], points['y'], points['z'])),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    element = PlyElement.describe(vertices, 'vertex')

    PlyData([element]).write(output_path)
    print(f"Point cloud saved to {output_path}")

def merge_plys(source_ply, target_ply, scale_objects, translation_x, translation_y, translation_z):
    source_vertices = source_ply['vertex'].data
    target_vertices = target_ply['vertex'].data
    print(target_vertices.shape)
    print(source_vertices.shape)
    cord = extract_coordinates(source_ply)

    new_vertices = np.zeros((len(target_vertices) + len(source_vertices)), dtype=target_dtype)

    x_min = target_vertices['x'].min()
    x_max = target_vertices['x'].max()
    y_min = target_vertices['y'].min()
    y_max = target_vertices['y'].max()
    z_min = target_vertices['z'].min()
    z_max = target_vertices['z'].max()

    print(f"x min: {x_min}, max: {x_max}")
    print(f"y min: {y_min}, max: {y_max}")
    print(f"z min: {z_min}, max: {z_max}")

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    # print(x_center, y_center, z_center)

    for field in target_dtype.names:
        if field in target_vertices.dtype.names:
            if field in ['x']:
                new_vertices[field][:len(target_vertices)] = (target_vertices[field]) 
            elif field in ['y']:
                new_vertices[field][:len(target_vertices)] = (target_vertices[field]) 
            elif field in ['z']:
                new_vertices[field][:len(target_vertices)] = (target_vertices[field]) 
            elif field.startswith('f_rest'):
                new_vertices[field][:len(target_vertices)] = target_vertices[field] 
            elif field.startswith('scale'):
                new_vertices[field][:len(target_vertices)] = target_vertices[field]+ np.log(2.5) # smooth the bg 
            else:
                new_vertices[field][:len(target_vertices)] = target_vertices[field]
        else:
            print(f"Property {field} is not included in the target PLY.")
            new_vertices[field][:len(target_vertices)] = 0.0

    x = new_vertices[:len(target_vertices)]['x']
    # print(np.min(x), np.max(x))
    y = new_vertices[:len(target_vertices)]['y']
    # print(np.min(y), np.max(y))
    z = new_vertices[:len(target_vertices)]['z']
    # print(np.min(z), np.max(z))

    for i, vertex in enumerate(source_vertices):
        for field in target_dtype.names:
            if field in vertex.dtype.names:
                if field in ['x', 'y', 'z']:
                    scaled_value = vertex[field]   
                    scale2 = scale_objects 
                   
                    if field == 'x':
                        new_vertices[field][len(target_vertices) + i] = scaled_value * scale2  
                    elif field == 'y':
                        new_vertices[field][len(target_vertices) + i] = scaled_value * scale2  
                    elif field == 'z':
                        new_vertices[field][len(target_vertices) + i] = scaled_value * scale2  
                elif field.startswith('scale'):
                    new_vertices[field][len(target_vertices) + i] = vertex[field] + np.log(scale2)
                else:
                    new_vertices[field][len(target_vertices) + i] = vertex[field]
            else:
                new_vertices[field][len(target_vertices) + i] = 0.0  

    for i in range(len(target_vertices), len(new_vertices['x'])):

        x = new_vertices['x'][i]
        y = new_vertices['y'][i]
        z = new_vertices['z'][i]
        
        # rotate the bg 3D to match with objects 
        # new_vertices['x'][i] = x
        # new_vertices['y'][i] = -y
        # new_vertices['z'][i] = -z   
        new_vertices['x'][i] += translation_x
        new_vertices['y'][i] += translation_y
        new_vertices['z'][i] += translation_z
       

    return new_vertices

def main(args):
    source_filename = f'./results/{args.task_name}/output.ply' # combined objects
    target_filename = f'./results/{args.task_name}/background/point_cloud.ply' # background
    output_filename = f'./results/{args.task_name}/point_cloud.ply'


    source_ply = read_ply_and_normalize(source_filename)
    target_ply = read_ply(target_filename)
    target_ply = rotate_points_180_x(target_ply) # rotate the bg 3D to match with objects 
  

    # mask_ply = read_ply('output.ply')
    # mask + target_ply

    #########################################################
    # 0. obtain the points of input view of the bg 3D model #
    #########################################################
    gaussian_ids = np.load(f"./results/{args.task_name}/visible_points.npy")
    print(gaussian_ids.shape)

    visible_points = extract_points_by_ids(target_ply, gaussian_ids)
    # print(visible_points.shape)

    save_points_to_ply_plyfile(visible_points, "visible_points.ply")

    # 1.1 Scale: calculate the range of x and y of the input view of the bg 3D model

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

    # meshes

    x_values = source_ply['vertex']['x']
    y_values = source_ply['vertex']['y']
    z_values = source_ply['vertex']['z']

    x_min_mesh, x_max_mesh = np.min(x_values), np.max(x_values)
    y_min_mesh, y_max_mesh = np.min(y_values), np.max(y_values)
    z_min_mesh, z_max_mesh = np.min(z_values), np.max(z_values)

    x_range_mesh = x_max_mesh - x_min_mesh 
    y_range_mesh = y_max_mesh - y_min_mesh
    z_range_mesh = z_max_mesh - z_min_mesh

    print("Combined objects:")
    print(f"x: min_mesh={x_min_mesh}, max_mesh={x_max_mesh}, range_mesh={x_range_mesh}")
    print(f"y: min_mesh={y_min_mesh}, max_mesh={y_max_mesh}, range_mesh={y_range_mesh}")
    print(f"z: min_mesh={z_min_mesh}, max_mesh={z_max_mesh}, range_mesh={z_range_mesh}")

    json_path = os.path.join("./results", args.task_name, "SAM", "label.json")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bboxes = [item['box'] for item in data['mask'] if 'box' in item]
    print(bboxes)

    big_bbox = get_bounding_box(bboxes)
   
    center_x = (big_bbox[0] + big_bbox[2]) / 2
    center_y = (big_bbox[1] + big_bbox[3]) / 2
    # print(center_x, center_y)
    ratio_center_x = center_x/1024
    ratio_center_y = center_y/1024
    print(ratio_center_x, ratio_center_y)

    ratio_x = (big_bbox[2] - big_bbox[0]) / 1024 
    ratio_y = (big_bbox[3] - big_bbox[1]) / 1024 
    print(ratio_x, ratio_y)

    scale_scene_x = x_range_mesh / (x_range * ratio_x) 
    scale_scene_y = y_range_mesh / (y_range * ratio_y)
    print(scale_scene_x, scale_scene_y)
    scale_scene = (scale_scene_x + scale_scene_y) / 2.0
    scale_scene = scale_scene_x
    print(scale_scene) 
    print("*********")
    print(f"x_range_scene: {x_range}, y_range_scene: {y_range}")
    ## calculate depth
    depth_array = np.load(f"results/{args.task_name}/depth/2DImage_pred.npy")

    mask_image = Image.open(f"results/{args.task_name}/background/mask_0.png").convert("L")
    mask_array = np.array(mask_image) 

    mask_array = (mask_array > 0).astype(np.uint8)  
    mask_depth_mean = np.mean(depth_array[mask_array == 1])

    non_mask_depth_mean = np.mean(depth_array[mask_array == 0])
    # depth ratio
    depth_ratio_objects_to_background = mask_depth_mean/non_mask_depth_mean
    scale_scene = scale_scene/depth_ratio_objects_to_background
    scale_objects = 1/scale_scene
    print(scale_scene, scale_objects) 
    
    depth_bg = np.load((f"results/{args.task_name}/depth/bg_pred.npy"))
    y_threshold = int(depth_bg.shape[0] * ratio_center_y)
    upper_part = np.mean(depth_bg[:y_threshold, :])
    lower_part = np.mean(depth_bg[y_threshold:, :])
    final_ratio_y = (ratio_center_y*upper_part)/(ratio_center_y*upper_part+(1-ratio_center_y)*lower_part) # compensation of the impact of depth on scale

    x_threshold = int(depth_bg.shape[1] * ratio_center_x)
    left_part = np.mean(depth_bg[:, :x_threshold])
    right_part = np.mean(depth_bg[:, x_threshold:])
    final_ratio_x = (ratio_center_x*left_part)/(ratio_center_x*left_part+(1-ratio_center_x)*right_part)
    print(final_ratio_x, final_ratio_y)

    x_min_scene = target_ply['vertex']['x'].min()
    x_max_scene = target_ply['vertex']['x'].max()
    y_min_scene = target_ply['vertex']['y'].min()
    y_max_scene = target_ply['vertex']['y'].max()
    z_min_scene = target_ply['vertex']['z'].min()
    z_max_scene = target_ply['vertex']['z'].max()

    print(f"x min: {x_min_scene}, max: {x_max_scene}")
    print(f"y min: {y_min_scene}, max: {y_max_scene}")
    print(f"z min: {z_min_scene}, max: {z_max_scene}")

    x_center = (x_min_scene + x_max_scene) / 2.0
    y_center = (y_min_scene + y_max_scene) / 2.0
    z_center = (z_min_scene + z_max_scene) / 2.0

    print(x_center, y_center, z_center) 

    # calculate the translation of x, y
    # center point of the big bbox
    # corresponding ratio of the start point (ratio_center_x/y)

    print(ratio_center_x, ratio_center_y) # image中objects的中点ratio
    # scale_scene = 1
    target_x = (x_min + final_ratio_x * x_range) # *scale_scene
    target_y = (-y_max + final_ratio_y * y_range) # *scale_scene # -y_max

    # source_x = x_min_mesh + ratio_center_x * x_range_mesh
    # source_y = y_min_mesh + ratio_center_y * y_range_mesh
    source_x = 0
    source_y = 0


    print(target_x-source_x, target_y-source_y)
    translation_x = target_x-source_x
    translation_y = target_y-source_y
    # calculate the corresponding scene point
    # subtract center multiply scale

    # z_axis
    # 500 points -> correspoinding z

    z_oris = np.load(f"results/{args.task_name}/z_oris.npy")
    z_targets = np.load(f"results/{args.task_name}/z_targets.npy")
    # print(np.mean(z_targets))

    nums_total = len(z_oris)
    # print(nums_total)
    # print(scale_objects)
    random_indices = random.sample(range(nums_total), 500)
    z_oris = z_oris[random_indices]

    z_targets = z_targets[random_indices]
    z_targets = - z_max + z_targets * z_range
    z_oris = z_oris * scale_objects
    z_moves = z_targets - z_oris

    filtered_z_moves, outlier_indices = remove_outliers_mad(z_moves)
    print(len(filtered_z_moves))
    print(np.mean(filtered_z_moves))
    translation_z = np.mean(filtered_z_moves)
    

    merged_vertices = merge_plys(source_ply, target_ply, scale_objects, translation_x, translation_y, translation_z)
    save_ply(output_filename, merged_vertices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Combination')
    parser.add_argument('--task_name', type=str, required=True, help='Task name.')

    args = parser.parse_args()
    main(args)

     