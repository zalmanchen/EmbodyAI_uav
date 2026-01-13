#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-7B 多 GPU 并行翻译脚本
✅ 利用 8 张 A100 同时处理，显著提升速度
"""

import os
import json
import glob
import gc


# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import torch
import copy
import argparse
import math
import multiprocessing as mp
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import AutoModelForVision2Seq, AutoProcessor

# ======================
# 🔧 全局配置
# ======================
DATASET_ROOT_PATH = "../data/openfly/traj"
TARGET_SIZE = (448, 448)
TEMP_IMAGE_DIR = "./tmp/qwen_vl_imgs"
MAX_NEW_TOKENS = 256  # 降低长度以加速

OUTPUT_DIR = "./train_t2rl_lora/data_v1"

BASE_MODEL_PATH = "./model/qwen/Qwen2.5-VL-7B-Instruct"

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
    for idx_str in index_list:
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

# ======================
# 🚀 优化的推理函数
# ======================
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
                use_cache=True,  # ✅ 启用 KV cache
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
# 🧵 多 GPU 工作进程
# ======================
def process_batch_on_gpu(gpu_id, batch_items, base_model_path, system_prompt, return_dict):
    """在指定 GPU 上处理一批样本"""
    print(f"GPU {gpu_id}: Starting to process {len(batch_items)} items")
    
    try:
        # 设置 CUDA 设备
        torch.cuda.set_device(gpu_id)
        
        # 加载模型到指定 GPU
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{gpu_id}",
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        processor.tokenizer.padding_side = "left"
        
        results = []
        for idx, item in enumerate(batch_items):
            try:
                weaken = item.get("weaken_instruction", "").strip()
                if not weaken:
                    continue
                    
                image_paths = load_image_paths(item)
                if not image_paths:
                    continue
                
                responses = inference_optimized(processor, model, image_paths, weaken, system_prompt)
                
                result_item = item.copy()
                result_item["translated_instruction_1"] = responses[0]
                result_item["translated_instruction_2"] = responses[1]
                result_item["translated_instruction_3"] = responses[2]
                result_item["translated_instruction_4"] = item.get("gpt_instruction", "").strip()
                results.append(result_item)
                
                if (idx + 1) % 10 == 0:
                    print(f"GPU {gpu_id}: Processed {idx + 1}/{len(batch_items)} items")
                    
            except Exception as e:
                print(f"GPU {gpu_id} error on item {idx}: {e}")
                continue
        
        print(f"GPU {gpu_id}: Completed {len(results)} items")
        return_dict[gpu_id] = results
        
    except Exception as e:
        print(f"GPU {gpu_id} fatal error: {e}")
        return_dict[gpu_id] = []

# ======================
# 🚀 主流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "eval"], required=True)
    parser.add_argument("--k_shot", type=int, default=20,
                        help="Few-shot: 随机采样 K 条数据（例如 5）")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="每条指令生成 N 个翻译（Few-Shot 建议 3）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（Few-Shot 采样可复现）")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="使用的 GPU 数量")
    args = parser.parse_args()

    # 加载数据
    input_json_path = {
        "train": "./train_t2rl_lora/data/t2rl_train_k20.json",
        "val": "./train_t2rl_lora/data/t2rl_val_k20.json",
        "eval": "./train_t2rl_lora/data/t2rl_eval_k20.json"
    }[args.split]

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    import random
    random.seed(args.seed)
    
    if args.k_shot is not None:
        if args.k_shot < len(data):
            selected_data = random.sample(data, args.k_shot)
        else:
            selected_data = data.copy()
            print(f"⚠️ Warning: Requested k_shot={args.k_shot} exceeds dataset size {len(data)}. Using full dataset.")
        
        data = selected_data
        print(f"📊 Few-Shot Sampling: Selected {len(data)} samples for k_shot={args.k_shot}")

    # 标准化数据格式
    if isinstance(data, dict):
        items = data.get('data', data.get('instructions', []))
    else:
        items = data

    print(f"🎯 Total items to process: {len(items)}")
    print(f"🎮 Using {args.num_gpus} GPUs")

    # 分割数据到多个 GPU
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    items_per_gpu = math.ceil(len(items) / num_gpus)
    
    batches = []
    for i in range(num_gpus):
        start_idx = i * items_per_gpu
        end_idx = min((i + 1) * items_per_gpu, len(items))
        if start_idx < len(items):
            batch = items[start_idx:end_idx]
            batches.append(batch)
            print(f"GPU {i}: assigned {len(batch)} items")
        else:
            batches.append([])

    # 创建多进程
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    base_model_path = BASE_MODEL_PATH
    system_prompt = build_system_prompt()

    for gpu_id in range(num_gpus):
        if len(batches[gpu_id]) > 0:
            p = mp.Process(
                target=process_batch_on_gpu,
                args=(gpu_id, batches[gpu_id], base_model_path, system_prompt, return_dict)
            )
            processes.append(p)
            p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并结果
    all_results = []
    total_processed = 0
    for gpu_id in range(num_gpus):
        if gpu_id in return_dict:
            gpu_results = return_dict[gpu_id]
            all_results.extend(gpu_results)
            total_processed += len(gpu_results)
            print(f"GPU {gpu_id}: contributed {len(gpu_results)} results")
    
    print(f"\n🎉 Total processed: {total_processed}/{len(items)} items")
    
    # 保存结果
    raw_save_path = os.path.join(OUTPUT_DIR, f"t2rl_{args.split}_k{args.k_shot}_n{args.n_samples}_with_translated.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(raw_save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Results saved to: {raw_save_path}")
    cleanup_temp_images()

if __name__ == "__main__":
    
    # 必须设置 spawn 启动方法
    mp.set_start_method('spawn', force=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    main()