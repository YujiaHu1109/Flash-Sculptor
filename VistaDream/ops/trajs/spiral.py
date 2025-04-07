import torch
import numpy as np

def generate_spiral_trajectory(num_frames, radius, forward_ratio=0.2, backward_ratio=0.8):
    t = np.linspace(0, 1, num_frames)
    r = np.sin(2 * np.pi * t) * radius
    # rotation angles at each frame
    theta = 2 * np.pi * t * num_frames 
    # try not to change y (up-down for floor and sky)
    x = r * np.cos(theta)
    y = r * np.sin(theta) * 0.3
    z = -r
    z[z<0]*=forward_ratio
    z[z>0]*=backward_ratio
    return x, y, z

def look_at(camera_position, target_position):
    # look at direction
    direction = target_position - camera_position
    direction /= np.linalg.norm(direction)
    # calculate rotation matrix
    up = np.array([0, 1, 0])
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    rotation_matrix = np.vstack([right, up, direction])
    rotation_matrix = np.linalg.inv(rotation_matrix)
    return rotation_matrix

def spiral_camera_poses(num_frames, radius, forward_ratio = 0.2, backward_ratio = 0.8, rotation_times = 0.3, look_at_times = 0.5):
    x, y, z = generate_spiral_trajectory(num_frames, radius*rotation_times, forward_ratio, backward_ratio)
    target_position = np.array([0,0,radius*look_at_times])
    camera_positions = np.vstack([x, y, z]).T
    camera_poses = []
    
    for pos in camera_positions:
        rotation_matrix = look_at(pos, target_position)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3,  3] = pos
        camera_poses.append(transform_matrix[None])
        
    camera_poses.reverse()
    camera_poses = np.concatenate(camera_poses,axis=0)
    
    return camera_poses

import torch
import numpy as np

def generate_circle_trajectory(num_frames, radius):
    """
    生成围绕y轴的圆形轨迹
    """
    theta = np.linspace(0, 2 * np.pi, num_frames)  # 均匀分布角度
    x = radius * np.cos(theta)  # x 坐标
    z = radius * np.sin(theta)  # z 坐标
    y = np.zeros_like(x)  # y 轴不变（保持水平）

    return x, y, z

def look_at(camera_position, target_position):
    """
    计算相机朝向目标点的旋转矩阵
    """
    direction = target_position - camera_position
    direction /= np.linalg.norm(direction)  # 归一化方向向量
    up = np.array([0, 1, 0])  # 竖直向上的参考向量
    right = np.cross(up, direction)  # 计算右方向向量
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)  # 重新计算正交的上方向

    rotation_matrix = np.vstack([right, up, direction])
    rotation_matrix = np.linalg.inv(rotation_matrix)  # 取逆得到旋转矩阵
    return rotation_matrix

def circular_camera_poses(num_frames, radius, look_at_height=0.5):
    """
    生成围绕目标旋转的相机位姿
    """
    x, y, z = generate_circle_trajectory(num_frames, radius)
    target_position = np.array([0, look_at_height, 0])  # 目标位置

    camera_positions = np.vstack([x, y, z]).T  # 形成相机位置数组
    camera_poses = []

    for pos in camera_positions:
        rotation_matrix = look_at(pos, target_position)  # 计算相机旋转矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = pos  # 赋值相机位置
        camera_poses.append(transform_matrix[None])  # 变成 (1, 4, 4) 方便拼接

    camera_poses = np.concatenate(camera_poses, axis=0)  # 合并所有位姿
    return camera_poses
