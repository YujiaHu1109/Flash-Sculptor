# Flash Sculptor: Modular 3D Worlds from Images

<br>

<a href="https://arxiv.org/abs/2504.06178"><img src="https://img.shields.io/badge/ariXv-2411.15098-A42C25.svg" alt="arXiv"></a>

> **Flash Sculptor: Modular 3D Worlds from Images**
> <br>
> Yujia Hu, 
> [Songhua Liu](http://121.37.94.87/), 
> [Xingyi Yang](https://adamdad.github.io/), 
> and 
> [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)
> <br>
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore
> <br>

![Demo](./assets/show.gif)

<strong>Abstract:</strong> Existing text-to-3D and image-to-3D models often struggle with complex scenes involving multiple objects and intricate interactions. Although some recent attempts have explored such compositional scenarios, they still require an extensive process of optimizing the entire layout, which is highly cumbersome if not infeasible at all. To overcome these challenges, we propose Flash Sculptor in this paper, a simple yet effective framework for compositional 3D scene/object reconstruction from a single image. At the heart of Flash Sculptor lies a divide-and-conquer strategy, which decouples compositional scene reconstruction into a sequence of sub-tasks, including handling the appearance, rotation, scale, and translation of each individual instance. Specifically, for rotation, we introduce a coarse-to-fine scheme that brings the best of both worlds--efficiency and accuracy--while for translation, we develop an outlier-removal-based algorithm that ensures robust and precise parameters in a single step, without any iterative optimization. Extensive experiments demonstrate that Flash Sculptor achieves at least a 3 times speedup over existing compositional 3D methods, while setting new benchmarks in compositional 3D reconstruction performance. 

<strong>Our Pipeline:</strong>

![Pipeline](./assets/teaser.jpg)

## 💻 Requirements
- Ubuntu 20.04
- CUDA 12.2
- Python 3.10.12
- Pytorch 2.4.0

## 🔧 Installation
For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## ⚙️ Pretrained Models
Please see [DOWNLOAD.md](DOWNLOAD.md) to download pretrained models.

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
cd Inpaint-Anything
python background_recover.py --task_name [task_name] --dilate_kernel_size 15 --lama_config ./lama/configs/prediction/default.yaml --lama_ckpt ./pretrained_models/big-lama
cd ..
```
Then, reconstruct the 3D scene of it using:
```bash
cd VistaDream
python vistadream.py --task_name [task_name]
cd ..
```

### 3. Depth estimation
```bash
cd ml-depth-pro
python run.py --task_name [task_name]
cd ..
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
cd ..
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
- [x] Release on arXiv.
- [ ] Improve codes to support images with resolutions other than (1024, 1024).
- [ ] Interactive demos.

## 🤔 Limitations
1. The result of segmentation may need manually adjusted if the segmented objects are not exactly what we want.
2. The inpainting module may occasionally produce suboptimal results.

## 💡 Citation
If you find this repo is helpful, please consider citing:
```
@article{hu2025flashsculptormodular3d,
  title={Flash Sculptor: Modular 3D Worlds from Objects},
  author={Yujia Hu and Songhua Liu and Xingyi Yang and Xinchao Wang},
  journal={arXiv preprint arXiv:2504.06178},
  year={2025}
}
```

## 🔗 Related Projects
We thank the excellent open-source projects:
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything.git) for the exceptional automatic segmentation performance;
- [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything.git) for the wonderful image inpainting performance;
- [VistaDream](https://github.com/WHU-USI3DV/VistaDream.git) for the efficient and fast 3D scene generation;
- [Depth-Pro](https://github.com/apple/ml-depth-pro) for accurate monocular depth estimation;
- [TRELLIS](https://github.com/microsoft/TRELLIS.git) for the high-fidelity and fast single-object 3D generation;
- [StableDiffusion](https://github.com/CompVis/stable-diffusion) for its powerful image generation and inpainting capabilities.
