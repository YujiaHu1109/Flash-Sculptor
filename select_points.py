# select points and obtain depth
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

def calculate_depth(x_cord, y_cord, depth_npy):

    x = np.arange(depth_npy.shape[1]) 
    y = np.arange(depth_npy.shape[0])  

    if depth_npy.size > 0:
        interp_func = interp2d(x, y, depth_npy, kind='linear')
        value = interp_func(x_cord, y_cord)[0]
        return value
        # average_depth = masked_depth.mean()
        # return average_depth
    else:
        print("No mask applied, or empty masked region.")


def select(args):
    masks_dir = os.path.join("results", args.task_name, "SAM")
    json_path = os.path.join("results", args.task_name, "SAM", "label.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    masks = {}
    for filename in os.listdir(masks_dir):
        if filename.endswith(".npy") and filename.startswith("bbox_mask_"):
            num_str = filename.split('_')[-1].split('.')[0]  
            num = int(num_str)
            print(num)

            mask = np.load(os.path.join(masks_dir, filename)) 
            masks[num] = mask  
 
    for i in range(len(masks)): 
        mask = masks[i]
        num = i
        white_coords = np.argwhere(mask)
        
        filtered_points = []
        neighborhood_range = 3

        for y, x in white_coords:
            is_inside = True
            for dy in range(-neighborhood_range, neighborhood_range + 1):
                for dx in range(-neighborhood_range, neighborhood_range + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        if not (mask[ny, nx] and
                                (ny > 0 and mask[ny-1, nx]) and
                                (ny < mask.shape[0] - 1 and mask[ny+1, nx]) and
                                (nx > 0 and mask[ny, nx-1]) and
                                (nx < mask.shape[1] - 1 and mask[ny, nx+1])):
                            is_inside = False
                            break
                if not is_inside:
                    break

            if is_inside:
                filtered_points.append((y, x))
        filtered_points = np.array(filtered_points)
        num_points_to_select = 500
        selected_coords = set()  

        edge_distance = 5  

        while len(selected_coords) < num_points_to_select:
            idx = np.random.choice(filtered_points.shape[0])
            point = filtered_points[idx]
  
            random_offset = np.random.uniform(-0.1, 0.1, size=2)  
            float_coord = point + random_offset
            if (edge_distance <= float_coord[0] < mask.shape[0] - edge_distance) and \
            (edge_distance <= float_coord[1] < mask.shape[1] - edge_distance):
                selected_coords.add(tuple(float_coord))
              
        float_coords = np.array(list(selected_coords))

        # plt.figure(figsize=(10, 10))
        # plt.imshow(mask, cmap='bone', interpolation='bilinear')
        # plt.scatter(float_coords[:, 1], float_coords[:, 0], 
        #             color='#ff4500', edgecolors='black', s=30, alpha=0.8, label='Selected Points')

        # plt.title('Selected Float Coordinates on Mask', fontsize=16, fontweight='bold')
        # plt.xlabel('Width', fontsize=14)
        # plt.ylabel('Height', fontsize=14)

        # plt.xticks([])
        # plt.yticks([])

        # plt.savefig(f"select{i}.png", dpi=300, bbox_inches='tight')

        points_cord = np.zeros((500, 3))
        points_cord[:, :2] = float_coords

        depth_data = np.load(os.path.join("results", args.task_name, "depth", "2DImage_pred.npy"))
        box = [item['box'] for item in data['mask'] if item['value'] == num+1]
        # print(box.shape)
        label = [item['label'] for item in data['mask'] if item['value'] == num+1]
        print(label)
        
        points_cord_save = points_cord.copy()
        
        points_cord_save[:, 0] = points_cord_save[:, 0]/(box[0][3] - box[0][1])
        points_cord_save[:, 1] = points_cord_save[:, 1]/(box[0][2] - box[0][0])

        points_cord[:, 0] += box[0][1]
        points_cord[:, 1] += box[0][0]

        for i in range(points_cord.shape[0]):
            x, y = points_cord[i, 0], points_cord[i, 1]
            points_cord[i, 2] = calculate_depth(y, x, depth_data)
            points_cord_save[i, 2] = points_cord[i, 2]
        npy_dir = os.path.join("results", args.task_name, "Single3DN")
        points_cord_save[:, [0, 1]] = points_cord_save[:, [1, 0]]
        np.save(os.path.join(npy_dir, f'points_cord_{num}.npy'), points_cord_save)
        


parser = argparse.ArgumentParser(description="Select points for depth alignment.")
parser.add_argument("--task_name", type=str, required=True, help="Task name.")

args = parser.parse_args()

if __name__ == "__main__":
    select(args)



