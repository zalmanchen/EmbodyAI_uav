#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-7B 官方兼容评估脚本 | 终极修复版
✅ 修复所有 grid_thw/unpack 错误 | ✅ 官方 chat 模式
"""

import os
import json
import glob
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import copy
import argparse
import pandas as pd
from PIL import Image
from io import BytesIO
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

# ======================
# 🔧 全局配置
# ======================
DATASET_ROOT_PATH = "../data/openfly/traj"
TARGET_SIZE = (448, 448)
TEMP_IMAGE_DIR = "./tmp/qwen_vl_imgs"
MAX_NEW_TOKENS = 256

OUTPUT_DIR = "./train_t2rl_lora/data_v3"

BASE_MODEL_PATH = "./model/qwen/Qwen2.5-VL-7B-Instruct_k100_v2" # _, v1, v2, v3, v4, v5


# ======================
# 🧼 初始化
# ======================
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

def cleanup_temp_images():
    files = glob.glob(os.path.join(TEMP_IMAGE_DIR, "openfly_*.jpg"))
    for f in files:
        try:
            os.remove(f)
        except:
            pass

import atexit
atexit.register(cleanup_temp_images)

# ======================
# 🖼️ 图像预处理
# ======================
def load_image_paths(item: dict) -> list:
    """加载图像路径"""
    item_id = item.get("id", "unknown_id")
    parquet_name = item.get("image_path")
    index_list = item.get("index_list", [])
    
    if not parquet_name or not index_list:
        return []

    parquet_path = os.path.join(DATASET_ROOT_PATH, f"{parquet_name.strip()}.parquet")
    try:
        df = pd.read_parquet(parquet_path)
    except:
        return []

    temp_img_paths = []
    for idx_str in index_list:  # 最多8张图（Qwen-VL 最佳实践）
        try:
            frame_idx = int(str(idx_str).split('_')[-1])
            img_bytes = df["image"][frame_idx]["bytes"]
            
            fname = f"openfly_{item_id}_{frame_idx:04d}.jpg"
            temp_path = os.path.join(TEMP_IMAGE_DIR, fname)
            
            with BytesIO(img_bytes) as buf:
                img = Image.open(buf).convert("RGB")
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                img.save(temp_path, "JPEG", quality=40)
            
            temp_img_paths.append({"path": temp_path})
        except:
            pass
    
    return temp_img_paths

# ======================
# 📝 系统提示词
# ======================
def build_system_prompt(agent_name: str = "openfly") -> str:
    role = (
        "You are an Intent-Driven Instruction Translator for UAV navigation.\n"
        "Your task is to convert vague human instructions into precise, executable commands "
        "that strictly follow the style of the target VLA agent. "
        "You will receive CONTINUOUS UAV trajectory frames (6~30+), follow the flight order strictly for translation."
    )
    few_shot = """
### Few-Shot Examples:
"Head directly toward the tall , light beige building with many windows . Then , slightly turn right and proceed to another large building characterized by its light gray color and balcony - like structures . Finish by slightly turning left , continuing straight towards a tall , multi - story skyscraper with large , beige windows featuring arched tops ."
"Proceed directly to the grey urban rooftop featuring antennas and equipment on a medium - sized building , then slightly turn left and head straight towards it ."
"Advance towards the gray skyscraper characterized by a tall building , then slightly turn right slightly and proceed to it . Finally , slightly turn left and continue straight to it ."
""".strip()
    constraints = (
        "\n### Critical Constraints (MUST FOLLOW FOR LONG TRAJECTORY):\n"
        "- STRICTLY ground all descriptions in the CONTINUOUS trajectory frames, follow the flight order completely.\n"
        "- DO NOT hallucinate objects, colors, structures or directions absent in the visuals.\n"
        "- PRESERVE the user's core intent and flight sequence, never change target order or invert directions.\n"
        "- Output fluent, multi-step, descriptive English that strictly matches the Few-Shot style, no redundant words.\n"
    )
    return f"{role}\n\n{few_shot}\n{constraints}"

STRATEGIES = [
    {"temp": 0.1, "top_p": 0.8, "rep_pen": 1.2, "name": "deterministic"},
    {"temp": 0.7, "top_p": 0.95, "rep_pen": 1.0, "name": "diverse"}, 
    {"temp": 1.0, "top_p": 1.0, "rep_pen": 1.0, "name": "max_random"}
]

from qwen_vl_utils import process_vision_info
def inference_optimized(processor, model, image_paths: list, instruction: str, system_prompt: str) -> list:
    """优化版本：只编码一次输入，但分别生成"""
    
    # 构造 messages
    content = []
    for p in image_paths:
        content.append({"type": "image", "image": p["path"]})
    content.append({"type": "text", "text": instruction})
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    
    # 只调用一次 process_vision_info 和 apply_chat_template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 只编码一次
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    responses = []
    STRATEGIES = [
        {"temp": 0.1, "top_p": 0.8, "rep_pen": 1.2},
        {"temp": 0.7, "top_p": 0.95, "rep_pen": 1.0}, 
        {"temp": 1.0, "top_p": 1.0, "rep_pen": 1.0}
    ]
    
    # 分别生成，但重用已编码的 inputs
    for cfg in STRATEGIES:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=cfg["temp"],
                top_p=cfg["top_p"],
                repetition_penalty=cfg["rep_pen"],
                use_cache=True,  # ✅ 启用 KV cache（在单次生成内有效）
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            gen_text = processor.decode(
                generated_ids[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            responses.append(gen_text)
    
    return responses

# ======================
# 🚀 主流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "eval"], required=True)
    parser.add_argument("--k_shot", type=int, default=100,
                        help="Few-shot: 随机采样 K 条数据（例如 5）")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="每条指令生成 N 个翻译（Few-Shot 建议 3）")
    # parser.add_argument("--disable_deterministic", action="store_true",
    #                     help="Few-Shot 训练必需：禁用确定性推理")
    parser.add_argument("--seed", type=int, default=42,  # ✅ 新增
                        help="随机种子（Few-Shot 采样可复现）")
    args = parser.parse_args()

    # 加载数据
    input_json_path = {
        "train": "./train_t2rl_lora/data/t2rl_train_k100.json",
        "val": "./train_t2rl_lora/data/t2rl_val_k100.json",
        "eval": "./train_t2rl_lora/data/t2rl_eval_k100.json"
    }[args.split]

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    import random

    if args.k_shot is not None:
        # setting seed for reproducibility
        random.seed(args.seed)
        if args.k_shot < len(data):
            selected_data = random.sample(data, args.k_shot)
        else:
            selected_data = data.copy()
            print(f"⚠️ Warning: Requested k_shot={args.k_shot} exceeds dataset size {len(data)}. Using full dataset.")
        
        data = selected_data
        print(f"📊 Few-Shot Sampling: Selected {len(data)} samples for k_shot={args.k_shot}")



    # 加载模型

    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    
    system_prompt = build_system_prompt()

    
    # 标准化
    if isinstance(data, dict):
        items = data.get('data', data.get('instructions', []))
        is_dict = True
        outer_key = 'data' if 'data' in data else 'instructions'
    else:
        items = data
        is_dict = False

    # 处理
    updated_items = []
    for i, item in enumerate(items):
        print(f"\n🖼️ Sample {i+1}/{len(items)}")
        
        weaken = item.get("weaken_instruction", "").strip()
        if not weaken:
            continue
            
        image_paths = load_image_paths(item)
        if not image_paths:
            continue
        

        response = inference_optimized(processor, model, image_paths, weaken, system_prompt)
        print(f"  ✅ {response[:80]}...")
        
        # 保存
        new_item = copy.deepcopy(item)
        new_item["translated_instruction_1"] = response[0]
        new_item["translated_instruction_2"] = response[1]
        new_item["translated_instruction_3"] = response[2]
        new_item["translated_instruction_4"] = item.get("gpt_instruction", "").strip()
        
        updated_items.append(new_item)
        
        # 清理
        gc.collect()
        torch.cuda.empty_cache()

    # 保存结果
    raw_save_path = os.path.join(OUTPUT_DIR, f"t2rl_{args.split}_k{args.k_shot}_n{args.n_samples}_with_translated.json")
    with open(raw_save_path, "w", encoding="utf-8") as f:
        json.dump(updated_items, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Done! Translated {len(updated_items)} samples")
    cleanup_temp_images()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TORCH_USE_CUDA_DSA"] = "0"
    # os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    torch.cuda.empty_cache()
    gc.collect()
    main()
