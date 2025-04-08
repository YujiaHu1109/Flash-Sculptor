import torch
import argparse
import os
import json
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting
from scipy.ndimage import binary_dilation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = 3407
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load the inpainting pipeline
    # pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    #     # "stabilityai/stable-diffusion-2-inpainting",
    #     "stabilityai/stable-diffusion-xl-inpainting",
    #     torch_dtype=torch.float16,
    # )
    pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

    pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()

    input_dir = os.path.join("./results", args.task_name, "Inpaint")
    output_dir = os.path.join("./results", args.task_name, "Single")
    json_path = os.path.join("./results", args.task_name, "SAM", "label.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_items = [
        {"value": item["value"], "label": item["label"]}
        for item in data["mask"]
        if item["value"] != 0
    ]

    for item in extracted_items:
        value = item["value"] - 1
        label = item["label"]
        print(f"Value: {value}, Label: {label}")

        init_image = Image.open(os.path.join(input_dir, f"bbox_image_{value}.jpg"))
        init_image = init_image.convert("RGB")
        init_image_np = np.array(init_image)

        original_size = init_image.size
        print(f"Original Image Size: {original_size}")

        mask = np.load(os.path.join(input_dir, f"object_{value}_with_occlusions.npy"))

        structure = np.ones((5, 5))
        dilated_mask = binary_dilation(mask, structure=structure)

        init_image = Image.fromarray(init_image_np)

        mask_image = dilated_mask.astype(np.uint8) * 255

        mask_image_pil = Image.fromarray(mask_image)

        # mask_image_pil.save("mask_image.png")
        # init_image.save("init_image_with_white_area.png")

        # Prepare prompts for inpainting
        prompt = f"a complete {label} with no other objects"
        negative_prompt = "any other objects"

        image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image_pil, guidance_scale=7.5).images[0]

        image = image.resize(original_size, Image.LANCZOS)

        image.save(os.path.join(output_dir, f"object_{value}.png"))
        print(f"Saved inpainted image at {os.path.join(output_dir, f'object_{value}.png')}")