import torch
import sys
import argparse
import os
import re
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points

def setup_args(parser):
    # parser.add_argument(
    #     "--input_img", type=str, required=True,
    #     help="Path to a single input img",
    # )
    # parser.add_argument(
    #     "--mask_npy", type=str, required=True,
    #     help="Path to the mask npy file",
    # )
    parser.add_argument(
        "--task_name", type=str, required=True,
        help="task_name",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    # parser.add_argument(
    #     "--output_dir", type=str, required=True,
    #     help="Output path to the directory with results.",
    # )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )

def load_npy_files_from_directory(directory):
    mask_files = [f for f in os.listdir(directory) if re.match(r'mask_\d+\.npy$', f)]

    masks = []
    
    for file in mask_files:
        mask = np.load(os.path.join(directory, file))
        print(mask.shape)
        if int(re.findall(r'\d+', file)[0]) == 6:
            pass
        else:
            masks.append(mask)
    
    combined_mask = np.any(masks, axis=0)  

    return combined_mask

def save_combined_mask(combined_mask):
    final_array = np.where(combined_mask, 255, 0).astype(np.uint8)
    
    return final_array

if __name__ == "__main__":
    """Example usage:
    python background_recover.py \
        --input_img ../0.jpg \
        --mask_npy ../outputs/0 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the input image
    args.input_img = os.path.join("../results", args.task_name, "2DImage.png")
    args.mask_npy = os.path.join("../results", args.task_name, "SAM")
    args.output_dir = os.path.join("../results", args.task_name, "background")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    img = load_img_to_array(args.input_img)

    # Load the mask from npy file
    # masks = np.load(args.mask_npy).astype(np.uint8) * 255
    combined_mask = load_npy_files_from_directory(args.mask_npy)

    masks = save_combined_mask(combined_mask)
    

    # Dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # Visualize and save the segmentation results
    img_stem = Path(args.input_img).stem
    # out_dir = Path(args.output_dir) / img_stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # Path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # Save the mask
        save_array_to_img(mask, mask_p)

        # Save the masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        # img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted_p = out_dir / f"background_recover.png"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)
