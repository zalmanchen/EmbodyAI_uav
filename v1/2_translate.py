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

OUTPUT_DIR = "./train_t2rl_lora/data"


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
# def build_system_prompt(agent_name: str = "openfly") -> str:
#     role = (
#         "You are an Intent-Driven Instruction Translator for UAV navigation.\n"
#         "Your task is to convert vague human instructions into precise, executable commands "
#         "that strictly follow the style of the target VLA agent. "
#         "You will receive CONTINUOUS UAV trajectory frames (6~30+), follow the flight order strictly for translation."
#     )
#     few_shot = """
# ### Few-Shot Examples (OpenFly Style):
# "Head directly toward the tall , light beige building with many windows . Then , slightly turn right and proceed to another large building characterized by its light gray color and balcony - like structures . Finish by slightly turning left , continuing straight towards a tall , multi - story skyscraper with large , beige windows featuring arched tops ."
# "Proceed directly to the grey urban rooftop featuring antennas and equipment on a medium - sized building , then slightly turn left and head straight towards it ."
# "Advance towards the gray skyscraper characterized by a tall building , then slightly turn right slightly and proceed to it . Finally , slightly turn left and continue straight to it ."
# """.strip()
#     constraints = (
#         "\n### Critical Constraints (MUST FOLLOW FOR LONG TRAJECTORY):\n"
#         "- STRICTLY ground all descriptions in the CONTINUOUS trajectory frames, follow the flight order completely.\n"
#         "- DO NOT hallucinate objects, colors, structures or directions absent in the visuals.\n"
#         "- PRESERVE the user's core intent and flight sequence, never change target order or invert directions.\n"
#         "- Output fluent, multi-step, descriptive English that strictly matches the Few-Shot style, no redundant words.\n"
#     )
#     return f"{role}\n\n{few_shot}\n{constraints}"

def build_system_prompt(agent_name: str = "openfly") -> str:
    role = (
    "You are a precision UAV instruction translator specialized in aerial navigation.\n"
    "Your task is to convert high-level human instructions into detailed, executable flight commands\n"
    "that maintain strict adherence to the observed visual trajectory and flight sequence."
)

    few_shot = """
    ### Navigation Command Style (Observe Pattern):
    "Head directly toward the tall, light beige building with many windows. Then, slightly turn right and proceed to another large building characterized by its light gray color and balcony-like structures. Finally, slightly turn left and continue straight towards a tall, multi-story skyscraper with large, beige windows featuring arched tops."
    "Proceed directly to the grey urban rooftop featuring antennas and equipment on a medium-sized building. Then, slightly turn left and head straight towards it."
    "Advance towards the gray skyscraper characterized by a tall building. Then, slightly turn right and proceed to it. Finally, slightly turn left and continue straight to it."
    """.strip()

    constraints = (
        "\n### Translation Requirements:\n"
        "• MAINTAIN flight order: preserve the exact sequence of targets\n"
        "• GROUND in visuals: describe only objects and features visible in the trajectory\n"
        "• NO hallucinations: omit objects, colors, or structures not present in frames\n"
        "\n### Output Specifications:\n"
        "- Single coherent paragraph\n"
        "- Complete sentences with proper punctuation\n"
        "- Professional technical language suitable for UAV operations"
    )
    return f"{role}\n\n{few_shot}\n{constraints}"


STRATEGIES = [
    {"temp": 0.2, "top_p": 0.9, "rep_pen": 1.0, "name": "conservative"},
    {"temp": 0.7, "top_p": 0.8, "rep_pen": 1.0, "name": "diverse"},
    {"temp": 0.4, "top_p": 0.95, "rep_pen": 1.2, "name": "anti-repeat"}
]


from qwen_vl_utils import process_vision_info




def inference(processor, model, image_paths: list, instruction: str, system_prompt: str) -> str:

    # 构造 messages
    content = []

    # 添加图像（路径字符串，processor 会自动加载）
    for p in image_paths:
        content.append({"type": "image", "image": p["path"]})

    content.append({"type": "text", "text": instruction})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # process_vision_info 内部会处理图像路径 以及 chunking 多图像
    image_inputs, video_inputs = process_vision_info(messages)

    # processing inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output = []
    for i in range(len(STRATEGIES)):
        cfg = STRATEGIES[i]
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=cfg["temp"],
                top_p=cfg["top_p"],
                repetition_penalty=cfg["rep_pen"],
                use_cache=False,
            )
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output.append(output_text[0].strip())

    return output

# ======================
# 🚀 主流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "eval"], required=True)
    parser.add_argument("--k_shot", type=int, default=5,
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
        "train": "./train_t2rl_lora/data/t2rl_train.json",
        "val": "./train_t2rl_lora/data/t2rl_val.json",
        "eval": "./train_t2rl_lora/data/t2rl_eval.json"
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
    base_model_path = "./model/qwen/Qwen2.5-VL-7B-Instruct"

    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to(device="cuda", dtype=torch.bfloat16)

    processor = AutoProcessor.from_pretrained(
        "./model/qwen/Qwen2.5-VL-7B-Instruct",
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
        

        response = inference(processor, model, image_paths, weaken, system_prompt)
        print(f"  ✅ {response[:80]}...")
        
        # 保存
        new_item = copy.deepcopy(item)
        new_item["translated_instruction_1"] = response[0]
        new_item["translated_instruction_2"] = response[1]
        new_item["translated_instruction_3"] = response[2]
        
        updated_items.append(new_item)
        
        # 清理
        gc.collect()
        torch.cuda.empty_cache()

    # 保存结果
    raw_save_path = os.path.join(OUTPUT_DIR, f"t2rl_{args.split}_with_translated.json")
    with open(raw_save_path, "w", encoding="utf-8") as f:
        json.dump(updated_items, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Done! Translated {len(updated_items)} samples")
    cleanup_temp_images()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "0"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    torch.cuda.empty_cache()
    gc.collect()
    main()
