# Flash Sculptor: Modular 3D Worlds from Images

> **Flash Sculptor: Modular 3D Worlds from Images**<br/>Yujia Hu, Songhua Liu, Xingyi Yang and Xinchao Wang

![Pipeline](./teaser.jpg)

## 💻 Requirements
- Ubuntu 20.04
- CUDA 12.2
- Python 3.10.12
- Pytorch 2.4.0

## 🔧 Installation
For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## 🔦 Run
Follow these steps to get a composite 3D scene from a single image:

### 0. Prepare an image
Obtain an image using the following command:
```bash
python t2i.py --task_name [task_name] --prompt [prompt]
```
Or you can simply put your own 2D image as `results/[task_name]/2DImage.png`.

### 1. Segment the image
Run the following command to segment the image and obtain the bounding box, mask and label of each object:
```bash
python segment.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --ram_checkpoint ram_swin_large_14m.pth --ram_plus_checkpoint ram_plus_swin_large_14m.pth --grounded_checkpoint groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_h_4b8939.pth --sam_hq_checkpoint sam_hq_vit_h.pth --box_threshold 0.25 --text_threshold 0.2 --iou_threshold 0.5 --device "cuda" --task_name [task_name]
```

### 2. Reconstruct the background scene
First, recover the background by running:
```bash
python background_recover.py --task_name [task_name] --dilate_kernel_size 15 --lama_config ./lama/configs/prediction/default.yaml --lama_ckpt ./pretrained_models/big-lama
```
Then, reconstruct the 3D scene of it using:
```bash
cd VistaDream
python vistadream.py --task_name [task_name]
```

### 3. Depth estimation
```bash
cd ml-depth-pro
python run.py --task_name [task_name]
```

### 4. Reconstruct single objects
First, inpaint the objects by:
```bash
python occlusion.py --task_name [task_name]
python inpaint.py --task_name [task_name]
```
Then, reconstruct the 3D point cloud of each object by:
```bash
cd TRELLIS
python trellis.py --task_name [task_name]
```

### 5. Combine the objects
First, determine the rotation by:
```bash
python rotation.py --task_name [task_name]
```
Then, select points for depth alignment:
```bash
python select_points.py	--task_name [task_name]
```
Finally, combine the objects together:
```bash
python combine_objects.py --task_name [task_name]
```

### 6. Combine with the background
```bash
python combine_scene.py --task_name [task_name]
```

## 🔦 ToDo List
- [ ] Release on arXiv.
- [ ] Improve README and files.
- [ ] Interactive demos.

## 🔗 Related Projects
We thank the excellent open - source projects:
- [Grounded - Segment - Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything.git) for the exceptional automatic segmentation performance;
- [Inpaint - Anything](https://github.com/geekyutao/Inpaint-Anything.git) for the wonderful image inpainting performance;
- [VistaDream](https://github.com/WHU-USI3DV/VistaDream.git) for the efficient and fast 3D scene generation;
- [Depth - Pro](https://github.com/apple/ml-depth-pro) for accurate monocular depth estimation;
- [TRELLIS](https://github.com/microsoft/TRELLIS.git) for the high - fidelity and fast single - object 3D generation;
- [StableDiffusion](https://github.com/CompVis/stable-diffusion) for its powerful image generation and inpainting capabilities.
