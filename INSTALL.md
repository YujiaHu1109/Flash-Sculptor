## INSTILLATION (Work in Progress)

### Requirements
We use an evironment with the following specifications, packages and dependencies:

- Ubuntu 20.04
- CUDA 12.1
- Python 3.10.12
- Pytorch 2.4.0

### Setup Instructions
- Basic environment
```bash
conda env create -f environment.yaml
conda activate flashsculptor
```

- Install the following packages based on your environment

```bash
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu120
```

- Install the following packages 

```bash
pip install git+https://github.com/EasternJournalist/utils3d
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git 
pip install /tmp/extensions/nvdiffrast
git clone https://github.com/autonomousvision/mip-splatting.git
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
```

- Install Depth-Pro

```bash
cd Vistadream/tools/DepthPro
pip install -e .
cd ../..
```

- Install requirements of OneFormer for scene reconstruction
```bash
# detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# MSDA
cd Vistadream/tools/OneFormer/oneformer/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../../..
```

