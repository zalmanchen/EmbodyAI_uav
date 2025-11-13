# 📁 airsim_utils.py

import airsim
import numpy as np
import os
import json
import time
from typing import Dict, Any, List, Tuple
import cv2 # 用于图像处理和保存

# 全局客户端实例
CLIENT_INSTANCE = None

def get_airsim_client():
    """初始化并返回 AirSim 客户端实例。"""
    global CLIENT_INSTANCE
    if CLIENT_INSTANCE is None:
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            print("AirSim Client Connection established.")
            CLIENT_INSTANCE = client
            CLIENT_INSTANCE.enableApiControl(True)
            CLIENT_INSTANCE.armDisarm(True)
            # 确保无人机处于悬停状态
            CLIENT_INSTANCE.moveByVelocityAsync(0, 0, 0, 1).join()
            print("Drone is ready.")
        except Exception as e:
            print(f"Error connecting to AirSim: {e}")
            raise
    return CLIENT_INSTANCE

def get_multimodal_data(client: airsim.MultirotorClient) -> Dict[str, Any]:
    """采集当前帧的 RGB、深度和语义分割图像。"""
    requests = [
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),      # RGB
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False), # Depth (Float)
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False) # Semantic
    ]

    responses = client.simGetImages(requests)
    data = {}
    
    # 解析图像数据
    for response in responses:
        if response.image_type == airsim.ImageType.Scene:
            img_rgb_bytes = response.image_data_uint8
            data["rgb_bytes"] = img_rgb_bytes
            data["img_dims"] = (response.height, response.width)
        elif response.image_type == airsim.ImageType.DepthPlanar:
            data["depth_array"] = airsim.get_pfm_array(response)
        elif response.image_type == airsim.ImageType.Segmentation:
            data["semantic_bytes"] = response.image_data_uint8
            
    # 获取姿态和动作
    kinematics = client.getMultirotorState().kinematics_estimated
    
    data["pose"] = {
        "position": [kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val],
        "orientation": [kinematics.orientation.w_val, kinematics.orientation.x_val, kinematics.orientation.y_val, kinematics.orientation.z_val]
    }
    
    # 动作向量（地面真值动作）
    data["action_vector"] = {
        "linear_velocity": [kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val],
        "angular_velocity": [kinematics.angular_velocity.z_val] # 仅 Yaw Rate
    }
    
    return data

def save_frame_data(base_path: str, traj_id: str, frame_id: int, data: Dict[str, Any]):
    """将采集到的数据保存到指定的路径，包括图像和元数据。"""
    
    frame_path = os.path.join(base_path, traj_id, str(frame_id).zfill(6))
    os.makedirs(frame_path, exist_ok=True)

    # 1. 图像保存 (需要额外的库如 cv2)
    img_rgb = np.frombuffer(data["rgb_bytes"], dtype=np.uint8).reshape(data["img_dims"][0], data["img_dims"][1], 3)
    cv2.imwrite(os.path.join(frame_path, "rgb.png"), img_rgb)
    
    # 2. 深度图保存 (使用 numpy 格式或 PFM 格式)
    np.save(os.path.join(frame_path, "depth.npy"), data["depth_array"])
    
    # 3. 元数据保存
    metadata = {
        "pose": data["pose"],
        "action_vector": data["action_vector"],
        "traj_id": traj_id,
        "frame_id": frame_id
    }
    with open(os.path.join(frame_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    return frame_path