import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
from scipy.interpolate import interp2d
import litellm
import pickle

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram, ram_plus
from ram import inference_ram
import torchvision.transforms as TS

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def check_tags_chinese(tags_chinese, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the tags_chinese if it is wrong. ' + \
                           f'tags_chinese: {tags_chinese}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised tags_chinese: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "tags_chinese: xxx, xxx, xxx"
        tags_chinese = reply.split(':')[-1].strip()
    return tags_chinese


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    pred_tags = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        pred_tags.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases, pred_tags


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)

def calculate_centroid_relative(bbox):

    centroid_x = (bbox[0] + bbox[2]) / 2
    centroid_y = (bbox[1] + bbox[3]) / 2

    return centroid_x/1024, centroid_y/1024, centroid_x.item(), centroid_y.item()

# def calculate_depth(x_cord, y_cord, depth_file):
#     assert mask.dim() == 3 and mask.size(0) == 1
#     depth_npy = np.load(depth_file)  

#     x = np.arange(depth_npy.shape[1]) 
#     y = np.arange(depth_npy.shape[0])  

#     if depth_npy.size > 0:
#         interp_func = interp2d(x, y, depth_npy, kind='linear')
#         value = interp_func(x_cord, y_cord)[0]
#         return value
#         # average_depth = masked_depth.mean()
#         # return average_depth
#     else:
#         print("No mask applied, or empty masked region.")

def extract_filename(image_path):
    filename_with_ext = image_path.split('/')[-1]  
    dot_index = filename_with_ext.rfind('.')
    if dot_index != -1:
        return filename_with_ext[:dot_index]
    return filename_with_ext  

def apply_mask_and_noise(image, mask, save_path):
    image_np = np.array(image)
    noise = np.random.normal(0, 255, image_np.shape).astype(np.uint8)
    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.where(mask_3c, image_np, noise)
    Image.fromarray(masked_image).save(save_path)


def apply_mask_and_noise_and_bbox(image, mask, bounding_box, save_path,):
    image_np = np.array(image)
    noise = np.full(image_np.shape, 255, dtype=np.uint8)
    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.where(mask_3c, image_np, noise)
    x0, y0, x1, y1 = map(int, bounding_box)
    bbox_region = masked_image[y0:y1, x0:x1]
    Image.fromarray(bbox_region).save(save_path)

def generate_inpaint_mask(image_path, mask, bounding_box, image_shape, save_path):
    inpaint_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = map(int, bounding_box)
    inpaint_mask[y0:y1, x0:x1] = 255
    inpaint_mask[mask] = 0

    cv2.imwrite(save_path, inpaint_mask)

    image_tmp, _ = load_image(image_path)
    bbox_image = image_tmp.crop((x0, y0, x1, y1))
    bbox_image_path = save_path.replace("inpaint_mask", "bbox_image")
    # cv2.imwrite(bbox_image_path, bbox_image)
    bbox_image.save(bbox_image_path)
    print(f"Bounding box image saved to {bbox_image_path}")

    bbox_mask = mask[y0:y1, x0:x1]
    bbox_mask_path = save_path.replace("inpaint_mask", "bbox_mask")
    np.save(bbox_mask_path, bbox_mask)
    print(f"Bounding box mask array saved to {bbox_mask_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything", add_help=True)
    parser.add_argument("--task_name", type=str, required=True, help="task name")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--ram_plus_checkpoint", type=str, default=None, help="path to ram++ checkpoint file"
    )
    parser.add_argument(
        "--use_ram_plus", action="store_true", help="using ram++ for prediction"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    task_name = args.task_name
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    ram_plus_checkpoint = args.ram_plus_checkpoint
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_ram_plus = args.use_ram_plus
    use_sam_hq = args.use_sam_hq
    image_path = os.path.join("results", task_name, "2DImage.png")
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    
    
    # ChatGPT or nltk is required when using tags_chineses
    # openai.api_key = openai_key
    # if openai_proxy:
        # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # make dir
    output_dir = os.path.join("results", task_name, "SAM")
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    
    # load model
    # initialize RAM
    if use_ram_plus:
        print("Initialize RAM-Plus Predictor")
        ram_model = ram_plus(pretrained=ram_plus_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    else:
        ram_model = ram(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()

    ram_model = ram_model.to(device)
    raw_image = image_pil.resize(
                    (384, 384))
    raw_image  = transform(raw_image).unsqueeze(0).to(device)

    res = inference_ram(raw_image , ram_model)

    # res_tmp = res[0]+" | swan"
    res_tmp = res[0]
    tags=res_tmp.replace(' |', ',')
    # tags = "alarm clock, clock, figurine, footstall, earth, miniature, stand, stool, toy, plant, glasses"
    print(tags)
  
    # tags=res[0].replace(' |', ',')
    tags_chinese=res[1].replace(' |', ',')

    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])

    # run grounding dino model
    boxes_filt, scores, pred_phrases, pred_tags = get_grounding_output(
        model, image, tags, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist() # a index list
    nms_idx = [0,2]
    print(nms_idx)
    boxes_filt = boxes_filt[nms_idx]
    # print(boxes_filt.shape)

    # pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    # pred_tags = [pred_tags[idx] for idx in nms_idx]
    pred_phrases_tmp = [pred_phrases[idx] for idx in nms_idx]
    pred_tags_tmp = [pred_tags[idx] for idx in nms_idx]
    # print(f"After NMS: {boxes_filt.shape[0]} boxes")
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    tags_chinese = check_tags_chinese(tags_chinese, pred_phrases_tmp)
    print(f"Revise tags_chinese with number: {tags_chinese}")

    # transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    
    with open('simple_pred_tags.pkl', 'wb') as f:
        pickle.dump(pred_tags_tmp, f)
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    print(masks.shape)

    # check boxes
    # boxes_final, masks_final, idxs = filter_bboxes_after_nms(boxes_filt, masks.cpu()) # torchSize([x, 4])
    boxes_final = boxes_filt
    masks_final = masks
    idxs = nms_idx
    print(boxes_final)

    # Determine objects to keep
    # Start:
    # indices_to_keep = [1,2,4,5,6]
    # idxs = [idxs[i] for i in indices_to_keep]
    # indices_to_keep = torch.tensor(indices_to_keep)

    # boxes_final = boxes_final[indices_to_keep]
    # masks_final = masks_final[indices_to_keep]
    # End

    print(boxes_final.shape, masks_final.shape, idxs)
    # idxs_final = [nms_idx[idx] for idx in idxs]
    idxs_final = idxs

    print(idxs_final)
    print(len(pred_phrases))
    pred_phrases = [pred_phrases[idx] for idx in idxs_final]
    pred_tags = [pred_tags[idx] for idx in idxs_final]
    # print(f"After NMS: {boxes_filt.shape[0]} boxes")
    print(f"After Checking: {boxes_final.shape[0]} boxes")
    print(pred_phrases, pred_tags)

    tags_chinese = check_tags_chinese(tags_chinese, pred_phrases)
    print(f"Revise tags_chinese with number: {tags_chinese}")
    
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    print(len(masks_final))
    translation = np.zeros((len(masks_final), 4)) # x, y, z, size
    for i, mask in enumerate(masks_final):
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        np.save(os.path.join(output_dir, f"mask_{i}.npy"), mask.cpu().numpy())
        # print(mask.shape)
        centroid = calculate_centroid_relative(boxes_final[i])

        # npy_path = os.path.join("results", task_name, "depth/2DImage_pred.npy")

        # print(calculate_depth(100, 1000, npy_path))
        # center_depth = calculate_depth(centroid[2], centroid[3], npy_path)
        translation[i][0] = centroid[0]
        translation[i][1] = centroid[1]
        translation[i][2] = 0
    np.save(os.path.join(output_dir, f"translation_{extract_filename(image_path)}.npy"), translation)

        
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)
    for box, label in zip(boxes_final, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    # plt.title('RAM-tags' + tags + '\n' + 'RAM-tags_chineseing: ' + tags_chinese + '\n')
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "automatic_label_output.jpg"), 
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    # save_mask_data(output_dir, tags_chinese, masks, boxes_filt, pred_phrases)
    save_mask_data(output_dir, tags_chinese, masks_final, boxes_final, pred_phrases)

    
    for idx, mask in enumerate(masks_final):
        save_path = os.path.join(output_dir, f"masked_noise_image_{idx}.jpg")
        apply_mask_and_noise(image, mask.cpu().numpy()[0], save_path)

    for idx, (mask, bounding_box) in enumerate(zip(masks_final, boxes_final)):
        save_path = os.path.join(output_dir, f"inpaint_mask_{idx}.jpg")
        save_path2 = os.path.join(output_dir, f"bbox_image_noise_{idx}.jpg")
        generate_inpaint_mask(image_path, mask.cpu().numpy()[0], bounding_box.cpu().numpy(), image.shape, save_path)
        apply_mask_and_noise_and_bbox(image, mask.cpu().numpy()[0], bounding_box.cpu().numpy(), save_path2)
      
