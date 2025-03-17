import numpy as np
import os
import json
import argparse

def load_bbox_and_masks(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bbox_list = []
    # tags = data["tags_chinese"].split(",")
    masks = data["mask"]

    for mask in masks:
        value = mask["value"] # 0: background, 1, 2, 3, 4, ...
        label = mask["label"] # teddy, box, ...
        if value == 0:
            pass
        else:
            box = mask["box"] # x0, y0, x1, y1
            bbox_list.append({
                "value": value,
                "label": label,
                "box": box
            })

    return bbox_list

def calculate_occlusions(bbox_list, depth_array_path, masks_dir, output_dir, trans_array_path):
    occlusions = {} 
    num_objects = len(bbox_list)
    print(num_objects)
    depth_array = np.load(depth_array_path)
    trans_array = np.load(trans_array_path)
    
    masks = []

    for i in range(num_objects):
        mask_path = os.path.join(masks_dir, f'mask_{i}.npy')
        if os.path.exists(mask_path):
            masks.append(np.load(mask_path))
        else:
            print(f"Warning: {mask_path} does not exist.")
            masks.append(np.zeros_like(depth_array)) 

    for i, bbox_info in enumerate(bbox_list):
        print(i)
        x_min, y_min, x_max, y_max = map(int, bbox_info['box'])
        max_length = max(x_max - x_min, y_max - y_min)
        trans_array[i][3] = max_length
        # print(x_min, y_min, x_max, y_max)

        current_mask = masks[i].squeeze(0)
        # print(current_mask)
        current_depth = depth_array[current_mask == True]
        print(current_depth.shape)

        current_max_depth = np.max(current_depth)
        print(current_max_depth)

        for j, other_mask in enumerate(masks):
            if i != j: 
                other_mask = other_mask.squeeze(0)
                other_mask_in_bbox = other_mask[y_min:y_max, x_min:x_max]
                
                if np.any(other_mask_in_bbox):
                    depth_bbox = depth_array[y_min:y_max, x_min:x_max]
                    other_depth = depth_bbox[other_mask_in_bbox == True]
                    other_min_depth = np.min(other_depth)
                    print(j, other_depth.shape, other_min_depth)
                    
                    if current_max_depth > other_min_depth:
                        if i not in occlusions:
                            occlusions[i] = []
                        occlusions[i].append(j)

        combined_mask = np.zeros((y_max-y_min, x_max-x_min), dtype=bool)
        
        print(combined_mask.shape)

        for occluded in occlusions.get(i, []):
            # print(occluded)    
            # print(combined_mask)
            combined_mask = combined_mask | masks[occluded].squeeze(0)[y_min:y_max, x_min:x_max]
            
        
        output_path = os.path.join(output_dir, f'object_{i}_with_occlusions.npy')
        np.save(output_path, combined_mask)

    np.save(trans_array_path, trans_array)
    return occlusions
def main(args):
    json_path = os.path.join('results', args.task_name, "SAM", "label.json")  
    masks_dir = os.path.join('results', args.task_name, "SAM") 
    output_dir = os.path.join('results', args.task_name, "Inpaint")
    depth_array_path = os.path.join('results', args.task_name, "depth",  "2DImage_pred.npy")  
    trans_array_path = os.path.join('results', args.task_name, "SAM", "translation_2DImage.npy")
    os.makedirs(output_dir, exist_ok=True)

    bbox_list = load_bbox_and_masks(json_path)
    occlusions = calculate_occlusions(bbox_list, depth_array_path, masks_dir, output_dir, trans_array_path)

    print("Occlusions:", occlusions)

    for obj_index, occluded_indices in occlusions.items():
        obj_label = bbox_list[obj_index]['label']
        occluded_labels = [bbox_list[j]['label'] for j in occluded_indices]
        print(f"Object '{obj_label}' is occluded by: {', '.join(occluded_labels)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask generation.")
    parser.add_argument('--task_name', type=str, required=True, help='task name') 
    # parser.add_argument('--path_dir', type=str, help='Directory containing images starting with mask_noise_')
    args = parser.parse_args()
    
    main(args)