from PIL import Image
import depth_pro
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Estimation')
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    args = parser.parse_args()

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    image_path = os.path.join("../results", args.task_name, "2DImage.png")
    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    # Normalize depth to range [0,1].
    depth = np.array(depth)
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    output_dir = os.path.join("../results", args.task_name, "depth")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "2DImage_pred.npy")

    np.save(output_path, normalized_depth)

    # Background.
    bg_path = os.path.join("../results", args.task_name, "background/background_recover.png")
    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(bg_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    # Normalize depth to range [0,1].
    depth = np.array(depth)
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    output_dir = os.path.join("../results", args.task_name, "depth")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "bg_pred.npy")

    np.save(output_path, normalized_depth)

    
    # depth_array_255 = (normalized_depth * 255).astype(np.uint8)
    # single_channel_image = Image.fromarray(depth_array_255, mode='L')

    # rgb_image = single_channel_image.convert('RGB')

    # rgb_image.save('output_rgb_image_pil.jpg')