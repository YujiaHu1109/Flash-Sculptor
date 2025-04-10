## DOWNLOAD WEIGHTS

### Download Weights for Grounded SAM
Grounding DINO
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
SAM
```bash
SAM:
vit_h:  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
vit_l:  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
vit_b:  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
SAM-HQ:
vit_h: https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing
vit_l: https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view?usp=sharing
vit_b: https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing
```
RAM
```bash
RAM++ (14M):    https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth 
RAM (14M):      https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth
Tag2Text (14M): https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/tag2text_swin_14m.pth
```

### Download Weights for Vistadream
To download pretrained models for [Fooocus](https://github.com/lllyasviel/Fooocus), [Depth-Pro](https://github.com/apple/ml-depth-pro), [OneFormer](https://github.com/SHI-Labs/OneFormer) and [SD-LCM](https://github.com/luosiallen/latent-consistency-model) to support Vistadream, run the following command:
```bash
cd Vistadream
bash download_weights.sh
```
