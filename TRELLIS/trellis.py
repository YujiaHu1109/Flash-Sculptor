import os
import argparse
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ['ATTN_BACKEND'] = 'xformers'
import imageio
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Reconstruction')
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    args = parser.parse_args()

    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    input_folder = os.path.join("../results", args.task_name, "Single")
    output_folder = os.path.join("../results", args.task_name, "Single3D")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image = Image.open(os.path.join(input_folder, image_file))
        # Run the pipeline
        outputs = pipeline.run(
            image,
            seed=1,
            # Optional parameters
            # sparse_structure_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            # slat_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 3,
            # },
        )
        # outputs is a dictionary containing generated 3D assets in different formats:
        # - outputs['gaussian']: a list of 3D Gaussians
        # - outputs['radiance_field']: a list of radiance fields
        # - outputs['mesh']: a list of meshes

        # Render the outputs
        # video = render_utils.render_video(outputs['gaussian'][0])['color']
        # imageio.mimsave("sample_gs.mp4", video, fps=30)
        # video = render_utils.render_video(outputs['radiance_field'][0])['color']
        # imageio.mimsave("sample_rf.mp4", video, fps=30)
        # video = render_utils.render_video(outputs['mesh'][0])['normal']
        # imageio.mimsave("sample_mesh.mp4", video, fps=30)

        # GLB files can be extracted from the outputs
        # glb = postprocessing_utils.to_glb(
        #     outputs['gaussian'][0],
        #     outputs['mesh'][0],
        #     # Optional parameters
        #     simplify=0.95,          # Ratio of triangles to remove in the simplification process
        #     texture_size=1024,      # Size of the texture used for the GLB
        # )
        # glb.export("sample.glb")

        # Save Gaussians as PLY files
        file_name, _ = os.path.splitext(image_file)
        output_ply_path = os.path.join(output_folder, f"{file_name}.ply")

        # Save Gaussians as PLY files
        outputs['gaussian'][0].save_ply(output_ply_path)

        print(f"Generated PLY file: {output_ply_path}")


