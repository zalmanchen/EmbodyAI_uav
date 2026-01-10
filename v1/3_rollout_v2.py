#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T²-RL Rollout Engine | 专为 Few-Shot 偏好对构建设计
✅ 核心修正：使用 translated_instruction 作为 VLA 输入
✅ 输出：logs/{split}_with_trajectory.json（供 build_preference_pairs.py 使用）
"""

import os
import json
import math
import time
import argparse
import subprocess
import threading
import numpy as np
import cv2
import torch
from PIL import Image
import airsim

# ========== 外部模型注册（保持不变） ==========
from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)

AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

# ========== 环境控制 ==========
def kill_env_process(keyword):
    result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
    cr_pid = result.stdout.decode().strip()
    if len(cr_pid) > 0:
        subprocess.run(['kill', '-9', cr_pid])

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(10)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

        # 轨迹指标
        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def _init_airsim_sim(self):
        env_dir = "envs/airsim/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(
            airsim.Vector3r(x, -y, -z),
            airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw))
        )
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def get_camera_data(self, camera_type='color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type: {camera_type}")

        image_type = {
            'color': airsim.ImageType.Scene,
            'depth': airsim.ImageType.DepthPlanar,
            'object_mask': airsim.ImageType.Segmentation
        }[camera_type]

        responses = self._client.simGetImages([
            airsim.ImageRequest('front_custom', image_type, False, False)
        ])
        response = responses[0]
        
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            return img_data.reshape(response.height, response.width)
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            return img_data.reshape(response.height, response.width, 3)

# ========== 动作处理 ==========
def get_images(lst, if_his, step):
    if not if_his:
        return lst[-1]
    if step == 1:
        return lst[-2:] if len(lst) >= 2 else [lst[0], lst[0]]
    if step == 2:
        return lst[-3:] if len(lst) >= 3 else [lst[0]] * (3 - len(lst)) + lst[:len(lst)]

def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0], dtype=np.float32),
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0], dtype=np.float32),
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=np.float32),
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.float32),
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0], dtype=np.float32),
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5], dtype=np.float32),
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }
    for idx, value in enumerate(action_dict.values()):
        if np.array_equal(action, value):
            return idx
    return 0

def get_action(policy, processor, image_list, instruction, acts, if_his=True, his_step=2):
    image_list = get_images(image_list, if_his, his_step)
    
    if isinstance(image_list, np.ndarray):
        images = [Image.fromarray(image_list)] * 3
    else:
        images = [Image.fromarray(img) for img in image_list]
    
    inputs = processor(instruction, images).to("cuda:0", dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    action = action.round().astype(int)
    return convert_to_action_id(action)

def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose
    step_size = 3.0

    if action == 0: pass
    elif action == 1: x += step_size * math.cos(yaw); y += step_size * math.sin(yaw)
    elif action == 2: yaw += math.radians(30)
    elif action == 3: yaw -= math.radians(30)
    elif action == 4: z += step_size
    elif action == 5: z -= step_size
    elif action == 6: x -= step_size * math.sin(yaw); y += step_size * math.cos(yaw)
    elif action == 7: x += step_size * math.sin(yaw); y -= step_size * math.cos(yaw)
    elif action == 8: x += step_size * math.cos(yaw) * 2; y += step_size * math.sin(yaw) * 2
    elif action == 9: x += step_size * math.cos(yaw) * 3; y += step_size * math.sin(yaw) * 3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return [x, y, z, yaw]

# ========== 核心评估逻辑 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/cx/Desktop/UAV/OpenFly/train_t2rl_lora/data/t2rl_train_with_translated.json", 
                        help="输入日志路径 (e.g., logs_train_k5_n3.json)")
    parser.add_argument("--output", type=str, default="./t2rl/data/rollout_with_trajectory.json",
                        help="输出带轨迹的日志路径")
    args = parser.parse_args()

    # 加载翻译结果
    with open(args.input, 'r') as f:
        all_eval_info = json.load(f)
    print(f"📦 Loaded {len(all_eval_info)} translations from {args.input}")
    
    # 加载 VLA Agent (固定 Proxy)
    model_path = "./model/openfly-agent"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    policy = AutoModelForVision2Seq.from_pretrained(
        model_path,
        #attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    print("✅ VLA Agent loaded (fixed Proxy)")

    # 初始化结果缓冲区
    rlaif_logs = []
    data_num = 0
    MAX_STEP = 100

    env_name = "env_airsim_18"
    print(f"\n🚀 Starting environment: {env_name} ({len(all_eval_info)} samples)")
    
    env_bridge = AirsimBridge(env_name)
    pos_ratio = 1.0  # AirSim 坐标缩放比例
    time.sleep(5)


    for idx, item in enumerate(all_eval_info):
        pos_list = item['pos']
        
        #### 需要 修正的地方：使用 translated_instruction 作为 VLA 输入！ ####

        new_item = item.copy()
        for i in range(4):
            acts = [] # reset action list for each sample
            data_num += 1

            if i==0:
                instruction_name = 'weaken_instruction' 
            else:
                instruction_name = f'translated_instruction_{i}'
            
            instruction = item[instruction_name]
            print(f"Processing sample {idx}, instruction variant {i}")  
            # 初始化无人机
            start_position = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_position[0], start_position[1], start_position[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_position} -> {end_position}, initial heading: {start_yaw}")
            
            stop_error = 1
            image_error = False

            # set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            env_bridge.set_camera_pose(
                start_position[0]/pos_ratio, 
                start_position[1]/pos_ratio, 
                start_position[2]/pos_ratio, 
                pitch, 
                np.rad2deg(start_yaw), 
                0
            )

            step = 0
            flag_osr = 0
            image_list = []
            env_bridge.pass_len = 1e-3
            old_pose = new_pose

            while step < MAX_STEP:
                try:
                    raw_image = env_bridge.get_camera_data()
                    cv2.imwrite("test/cur_img.jpg", raw_image)
                    image = raw_image
                    
                    image_list.append(image)
                    model_action = get_action(policy, processor, image_list, instruction, acts, if_his=True, his_step=2)
                    acts.append(model_action)
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    env_bridge.set_camera_pose(
                        new_pose[0]/pos_ratio, 
                        new_pose[1]/pos_ratio, 
                        new_pose[2]/pos_ratio, 
                        pitch, 
                        np.rad2deg(new_pose[3]), 
                        0
                    )
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = new_pose

                    if model_action == 0:
                        stop_error = 0
                        break
                    step += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break


            # 计算指标
            final_pos = new_pose[:3]
            dis = calculate_distance(end_position, final_pos)
            traj_len = calculate_distance(start_position, end_position)
            success = 1 if dis < 20 else 0
            spl = traj_len / env_bridge.pass_len if success else 0.0
            reward = 0.5*success + 0.3*spl + 0.2*math.exp(-0.1*dis/max(traj_len, 1e-3))

            new_item[f'{instruction_name}_reward'] = reward
            new_item[f'{instruction_name}_success'] = success
            new_item[f'{instruction_name}_spl'] = spl

            if image_error:
                continue

        
        # 保存完整日志
        rlaif_logs.append(new_item)

        print(f"✅ Result: SR={success}, SPL={spl:.3f}, Reward={reward:.3f}")

            

    # 清理环境
    kill_env_process("AirVLN")
    del env_bridge
    torch.cuda.empty_cache()

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rlaif_logs, f, indent=2)
    
    # 最终统计
    # sr = sum(r["success"] for r in rlaif_logs) / len(rlaif_logs)
    # avg_reward = sum(r["reward"] for r in rlaif_logs) / len(rlaif_logs)
    
    # print(f"\n✅ Rollout complete!")
    # print(f"   - Total samples: {len(rlaif_logs)}")
    # print(f"   - Avg Success Rate: {sr:.2%}")
    # print(f"   - Avg Reward: {avg_reward:.3f}")
    # print(f"   - Output saved to: {args.output}")

if __name__ == '__main__':
    main()