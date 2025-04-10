import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
import os
import copy
import imageio
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.optimize import minimize
from trellis.representations.gaussian.gaussian_model import Gaussian 
from trellis.utils import render_utils 

def get_angles_from_index(index, interval):
    y_steps = 360 // interval
    x_step = index // y_steps
    y_step = index % y_steps
    x_angle = x_step * interval
    y_angle = y_step * interval
    return x_angle, y_angle

def remove_white_borders(img):
    img_array = np.array(img)
    rows, cols = np.where(np.any(img_array < 255, axis=-1))
    if len(rows) == 0 or len(cols) == 0:
        return img
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    side_length = max(max_row - min_row, max_col - min_col)
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    half_side = side_length // 2
    new_min_row = max(0, center_row - half_side)
    new_max_row = min(img_array.shape[0], center_row + half_side)
    new_min_col = max(0, center_col - half_side)
    new_max_col = min(img_array.shape[1], center_col + half_side)
    cropped_img = img.crop((new_min_col, new_min_row, new_max_col, new_max_row))
    return cropped_img

def preprocess_image(img, device="cuda", target_size=224):
    img = remove_white_borders(img)
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

def get_feature_vector(model, img):
    with torch.no_grad():
        return model(img).squeeze() 

def compute_similarity(ref_feature, image_features):
    ref_feature = ref_feature / ref_feature.norm()  
    image_features = image_features / image_features.norm(dim=1, keepdim=True)  
    similarities = torch.mm(ref_feature.unsqueeze(0), image_features.T) 
    return similarities.squeeze()

def find_top_similar_images(images, yaws, pitchs, ref_image_path, dino_model, args):
    ref_image = Image.open(ref_image_path).convert("RGB")
    ref_tensor = preprocess_image(ref_image, args.device)
    ref_feature = get_feature_vector(dino_model, ref_tensor)

    image_tensors = torch.stack([preprocess_image(Image.fromarray(img), args.device) for img in images])
    image_features = torch.stack([get_feature_vector(dino_model, img) for img in image_tensors])

    similarities = compute_similarity(ref_feature, image_features)

    top_indices = similarities.topk(3).indices.cpu().numpy()

    for i, idx in enumerate(top_indices):
        idx  # This is the image index in the saved image list
        x, y = get_angles_from_index(idx, args.interval)
        print(f"Top {i+1} match: Saved Image Index {idx}, "
              f"x: {x}°, y: {y}°, Similarity: {similarities[idx].item():.6f}")

    return top_indices, ref_feature

def load_ply_as_gaussian(ply_path):
    gaussian_obj = Gaussian(
    aabb=[0, 0, 0, 1, 1, 1],  
    sh_degree=0,  
    device='cuda' 
    )

    gaussian_obj.load_ply(ply_path, transform=None)
    return gaussian_obj

def rotate_and_render(input_ply, args):
    image_list = []
    for x_angle in range(0, 360, args.interval):  
        for y_angle in range(0, 360, args.interval): 
            gaussian_points = load_ply_as_gaussian(input_ply)
            gaussian_points.rotate_around_x_axis(90+x_angle)
            gaussian_points.rotate_around_y_axis(180+y_angle)
            outputs = {'gaussian': [gaussian_points]}
            image, extrinsics, intrinsics = render_utils.render_single(outputs['gaussian'][0])
            image_list.append(image[0])
            # pil_image = Image.fromarray(image[0])
            # pil_image.save("Image.png")
            # exit(0)
    return image_list 

def save_images(image_list, output_dir="rendered_images"):
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(image_list):
        filename = os.path.join(output_dir, f"image_{i:03d}.png")  
        imageio.imwrite(filename, img) 
        # print(f"Saved: {filename}")

def objective_function(angles, gaussian_obj, dino_model, ref_feature):
    x_angle, y_angle = angles

    gaussian_copy = copy.deepcopy(gaussian_obj)
    # gaussian_rot = apply_rotation_to_gaussian(gaussian_copy, x_angle, y_angle)
    gaussian_copy.rotate_around_x_axis(x_angle)
    gaussian_copy.rotate_around_y_axis(180+y_angle)

    outputs = {'gaussian': [gaussian_copy]}
    images, extrinsics, intrinsics = render_utils.render_single(outputs['gaussian'][0])

    rendered_img = images[0]
    rendered_img_pil = Image.fromarray(rendered_img)

    img_tensor = preprocess_image(rendered_img_pil)
    img_feature = get_feature_vector(dino_model, img_tensor)

    similarity = compute_similarity(ref_feature, img_feature.unsqueeze(0))
    print(f"Testing angles x:{x_angle:.2f}, y:{y_angle:.2f}, similarity: {similarity.item():.4f}")
    return -similarity.item()

def rotate_x(coords, angle):
    """ Rotates coordinates around the X-axis by a given angle (in degrees) """
    rad = np.radians(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(rad), -np.sin(rad)],
                                [0, np.sin(rad), np.cos(rad)]])
    return np.dot(coords, rotation_matrix.T)

def rotate_y(coords, angle):
    """ Rotates coordinates around the Y-axis by a given angle (in degrees) """
    rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(rad), 0, np.sin(rad)],
                                [0, 1, 0],
                                [-np.sin(rad), 0, np.cos(rad)]])
    return np.dot(coords, rotation_matrix.T)


def rotate_ply(original_ply, output_ply_path, x_angle, y_angle):
    """ Reads a PLY file and rotates it around X, Y, and Z axes """
    # Read the PLY file
    ply_data = PlyData.read(original_ply)
    vertices = ply_data['vertex'].data  # Get the vertex data

    # Extract the original x, y, z coordinates
    coords = np.vstack((vertices['x'], vertices['y'], vertices['z'])).T

    # Apply rotations
    coords_rotated = rotate_x(coords, x_angle)
    coords_rotated = rotate_y(coords_rotated, y_angle)

    # Create new vertex data
    new_vertices = vertices.copy()
    new_vertices['x'] = coords_rotated[:, 0]
    new_vertices['y'] = coords_rotated[:, 1]
    new_vertices['z'] = coords_rotated[:, 2]
   
    new_ply_data = PlyData([PlyElement.describe(new_vertices, 'vertex')], text=ply_data.text)
    new_ply_data.write(output_ply_path)
    print(f"Rotate PLY file {original_ply} with X: {x_angle}°, Y: {y_angle}° saved to {output_ply_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--task_name", type=str, required=True, help="task name")
    parser.add_argument("--interval", type=int, default=15, help="interval angle")

    args = parser.parse_args()

    input_folder = os.path.join("../results", args.task_name, "Single3D")
    ref_folder = os.path.join("../results", args.task_name, "Single")
    output_folder = os.path.join("../results", args.task_name, "Single3DN")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ply_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('ply'))]
    

    for ply_file in ply_files:
        # rotate and render images
        input_ply = ply_file
        image_list = rotate_and_render(input_ply, args)
        # save_images(image_list)
        # print(len(image_list))

        # coarse search
        yaws = np.arange(0, 360, args.interval) 
        pitchs = np.arange(0, 360, args.interval)  
        yaw_grid, pitch_grid = np.meshgrid(yaws, pitchs, indexing='ij')
        yaw_list = yaw_grid.flatten()
        pitch_list = pitch_grid.flatten()

        ply_file = os.path.basename(ply_file)
        ply_name = os.path.splitext(ply_file)[0]
        ref_image_name = ply_name + ".png"
        ref_image_path = os.path.join(ref_folder, ref_image_name)
        print(ref_image_path)

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg").to(args.device).eval()

        top_indices, ref_feature = find_top_similar_images(image_list, yaw_list, pitch_list, ref_image_path, dino_model, args)
        x, y = get_angles_from_index(top_indices[0], args.interval)
        initial_guess = [x, y]

        gaussian_obj = load_ply_as_gaussian(input_ply)

        result = minimize(
            objective_function,
            initial_guess,
            args=(gaussian_obj, dino_model, ref_feature),
            method='Nelder-Mead',
            options={'xatol': 1e-2, 'fatol': 1e-2, 'disp': True}
        )

        optimal_angles = result.x
        print("Optimal rotation angles found (x_angle, y_angle):", optimal_angles)
        x_final, y_final = optimal_angles[0], optimal_angles[1]
        
        # rotate the ply file
        output_ply_path = os.path.join(output_folder, os.path.basename(input_ply))
        print(input_ply, output_ply_path)
        rotate_ply(input_ply, output_ply_path, x_final, y_final)


if __name__ == '__main__':
    main()

