import torch
from diffusers import StableDiffusion3Pipeline, EulerDiscreteScheduler, DDPMScheduler
import argparse
import os

def run_sd3_t2i(args):
    torch.manual_seed(args.seed)
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # toy_scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler")
    # pipe.scheduler = toy_scheduler
    image = pipe(
        args.prompt,
        # "A little girl in an oil painting style",
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        # latents = torch.rand(1,16,64,64).cuda()
        width=args.width,   
        height=args.height
    ).images[0]
    print(image.size)
    save_dir = os.path.join(args.outputs_dir, args.task_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "2DImage.png")
    image.save(save_path)
    print(f"Saved file to {save_path}")

parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
parser.add_argument("--prompt", type=str, required=True, help="The prompt for image generation.")
parser.add_argument("--outputs_dir", type=str, default="results", help="Save path.")
parser.add_argument("--task_name", type=str, required=True, help="Task name.")
parser.add_argument("--negative_prompt", type=str, default="", help="The negative prompt for image generation.")
parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps.")
parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale.")
parser.add_argument("--width", type=int, default=1024, help="Image width.")
parser.add_argument("--height", type=int, default=1024, help="Image height.")
parser.add_argument("--seed", type=int, default=666, help="Image height.")
parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")

args = parser.parse_args()

if __name__ == "__main__":
    run_sd3_t2i(args)

