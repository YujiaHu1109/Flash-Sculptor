# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import torch
import random
import argparse
import numpy as np
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, task):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            task = f"The task is {task}"

            inputs = {"image": image, "height": height, "width": width, "task": task}
            predictions = self.model([inputs])[0]
            return predictions

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/mnt/proj/SOTAs/OneFormer-main/configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", default='semantic',help="Task type")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.IS_DEMO', True,
                 'MODEL.IS_TRAIN', False,
                 'MODEL.WEIGHTS', '/mnt/proj/SOTAs/OneFormer-main/ckpts/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser

class OneFormer_Segment():
    def __init__(self,
                 ckpt='/mnt/proj/GaussianAnythingV2/Tools/OneFormer/ckpts/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth',
                 yaml='/mnt/proj/GaussianAnythingV2/Tools/OneFormer/configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml') -> None:
        self.ckpt = ckpt
        self.yaml = yaml
        self._setup()
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST_PANOPTIC[0] if len(self.cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        if 'cityscapes_fine_sem_seg_val' in self.cfg.DATASETS.TEST_PANOPTIC[0]:
            from cityscapesscripts.helpers.labels import labels
            stuff_colors = [k.color for k in labels if k.trainId != 255]
            self.metadata = self.metadata.set(stuff_colors=stuff_colors)
        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(self.cfg)

    def _setup(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args = get_parser().parse_args()
        args.config_file=self.yaml
        args.opts[-1] = self.ckpt
        self.cfg = setup_cfg(args)
        
    def __call__(self, img):
        if np.amax(img) < 2: img = img*255
        predictions = self.predictor(img, 'semantic')
        result = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
        sky_mask = result == 2
        return sky_mask
    
if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    runnor = OneFormer_Segment()
    image = Image.open(f'/mnt/proj/GaussianAnythingV2/GAv2/outpaint.png')
    image = np.array(image)
    mask=runnor(image)
    mask = (mask*255).astype(np.uint8)
    Image.fromarray(mask).save('mask.png')