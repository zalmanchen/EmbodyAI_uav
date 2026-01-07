#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T²-RL LoRA Trainer | 调试版 - 先确认 trajectory 格式
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO

TEMP_IMAGE_DIR = "./tmp/train_imgs"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

def load_rollout_logs(log_path: str):
    with open(log_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def save_image_to_temp_from_array(img: Image.Image, item_id: str, idx: int) -> str | None:
    try:
        fname = f"train_{item_id}_{idx:04d}.jpg"
        temp_path = os.path.join(TEMP_IMAGE_DIR, fname)
        img.resize((448, 448), Image.Resampling.LANCZOS).save(temp_path, "JPEG", quality=40)
        return temp_path
    except:
        return None

def debug_trajectory_format(log_path: str):
    """调试 trajectory 格式"""
    logs = load_rollout_logs(log_path)
    print(f"✅ 加载 {len(logs)} 个样本")
    
    for i, log in enumerate(logs[:3]):  # 只看前3个
        print(f"\n🔍 样本 {i}: {log.get('id', 'N/A')}")
        print(f"   Keys: {list(log.keys())}")
        
        traj = log.get("trajectory", [])
        print(f"   Trajectory 长度: {len(traj)}")
        
        if traj:
            step0 = traj[0]
            print(f"   首步 keys: {list(step0.keys())}")
            
            # 查找图像字段
            img_fields = []
            for k, v in step0.items():
                if isinstance(v, (np.ndarray, list)) and hasattr(v, '__len__'):
                    if hasattr(v, 'shape'):
                        shape = v.shape
                    else:
                        shape = np.array(v).shape if len(v) > 0 else 'unknown'
                    if len(shape) >= 2:  # 可能是图像
                        img_fields.append((k, shape))
            
            print(f"   图像候选: {img_fields}")
            
            # 尝试保存首帧图像
            if img_fields:
                k, shape = img_fields[0]
                try:
                    img_data = step0[k]
                    if isinstance(img_data, list):
                        img_data = np.array(img_data)
                    if img_data.dtype == np.float32 and img_data.max() <= 1.0:
                        img_data = (img_data * 255).astype(np.uint8)
                    
                    img = Image.fromarray(img_data)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    temp_path = save_image_to_temp_from_array(img, log.get("id", f"sample_{i}"), 0)
                    print(f"   ✅ 首帧图像保存至: {temp_path}")
                    
                except Exception as e:
                    print(f"   ❌ 图像保存失败: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python debug_traj.py <log_path>")
        sys.exit(1)
    
    debug_trajectory_format(sys.argv[1])