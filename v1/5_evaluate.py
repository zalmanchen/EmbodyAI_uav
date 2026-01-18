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

EVAL_JSON_PATH = "./train_t2rl_lora/data/t2rl_eval.json"
OUTPUT_JSON_PATH = "./train_t2rl_lora/data_v3/t2rl_eval_with_translated_k100.json" # v1, v2, v3, v4, v5
LORA_PATH = "/mnt/geogpt-doc-new/default/cx/UAV/OpenFly/train_t2rl_lora/data_v1/output/v2/baseline_dpo/final"

base_model_path ="/mnt/geogpt-doc-new/default/cx/UAV/OpenFly/model/qwen/Qwen2.5-VL-7B-Instruct_k100_v3" # v1 ,v2, v3, v4, v5

save_model_path = "/mnt/geogpt-doc-new/default/cx/UAV/OpenFly/model/qwen/Qwen2.5-VL-7B-Instruct_test" # save the updated model

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
    for idx_str in index_list[:8]:  # 最多8张图（Qwen-VL 最佳实践）
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
### Few-Shot Examples :
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
# 🤖 模型加载（✅ 关键修复：inference_mode=True + FA2）
# ======================

def manual_merge_lora(base_model, lora_path):
    import safetensors.torch
    weights = safetensors.torch.load_file(
        os.path.join(lora_path, "adapter_model.safetensors")
    )
    
    print("🔧 Manual merging LoRA weights...")
    state_dict = base_model.state_dict()
    merged_count = 0
    
    for name, lora_weight in weights.items():
        if "lora_A.weight" in name:  # 注意：你的键是 .lora_A.weight（无 .default）
            # 🔑 关键修复：双重 model → 单 model
            # 示例: "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
            # 目标: "model.language_model.layers.0.self_attn.q_proj.weight"
            base_name = name.replace(".lora_A.weight", ".weight")
            base_name = base_name.replace("base_model.model.model.", "model.")  # 双 model → 单 model
            
            # 获取 A 和 B
            A = lora_weight
            B_name = name.replace("lora_A", "lora_B")
            if B_name not in weights:
                print(f"⚠️ Missing B: {B_name}")
                continue
            B = weights[B_name]
            
            # 计算 delta_W
            delta = (B @ A) * 2.0  # alpha/r = 8/4
            
            if base_name in state_dict:
                state_dict[base_name] += delta.to(state_dict[base_name].dtype)
                merged_count += 1
                if merged_count <= 3:
                    print(f"  ✅ Merged: {base_name} | delta_norm={delta.norm().item():.2e}")
            else:
                # 🔍 调试：打印未匹配的键
                print(f"  ⚠️ Not found: {base_name}")
                # 尝试备选方案（移除所有 model. 前缀）
                alt_name = base_name.replace("model.", "", 1)
                if alt_name in state_dict:
                    state_dict[alt_name] += delta.to(state_dict[alt_name].dtype)
                    merged_count += 1
                    print(f"  ✅ Alt merged: {alt_name}")
    
    print(f"✅ Manual merge completed | {merged_count} modules updated")
    base_model.load_state_dict(state_dict)
    return base_model

def merge_lora(base_model_path: str, lora_path: str) -> AutoModelForVision2Seq:
    print(f"🔍 Loading base model on CPU...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    # 手动合并
    merged_model = manual_merge_lora(base_model, lora_path)
    
    # 🔑 关键修复：重新加载 base_model 用于验证（或直接用 merged_model 对比）
    print("🔍 Reloading base model for verification...")
    base_model_for_check = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    # ✅ 正确验证：merged_model vs fresh base_model
    layer_idx = min(20, len(merged_model.language_model.layers) - 1)
    base_w = base_model_for_check.language_model.layers[layer_idx].self_attn.q_proj.weight
    merged_w = merged_model.language_model.layers[layer_idx].self_attn.q_proj.weight
    max_diff = torch.abs(base_w - merged_w).max().item()
    print(f"✅ Final verification | Max diff: {max_diff:.6f} {'🟢 SUCCESS' if max_diff > 1e-5 else '🔴 FAILED'}")

    del base_model_for_check

    # 保存模型和配置

    merged_model.save_pretrained(save_model_path, safe_serialization=True)
    
    # 复制 processor 配置
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(save_model_path)

    return merged_model.to(device="cuda", dtype=torch.bfloat16)

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

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512, # 多图可能需要更长输出? 256不够?
            do_sample=False,
            use_cache=True,
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

# ======================
# 🚀 主流程
# ======================
if __name__ == "__main__":
    # 加载模型
    # model = merge_lora(
    #     base_model_path="./model/qwen/Qwen2.5-VL-7B-Instruct_v1",
    #     lora_path=LORA_PATH
    # )
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    
    system_prompt = build_system_prompt()

    # 加载数据
    with open(EVAL_JSON_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 标准化
    if isinstance(eval_data, dict):
        items = eval_data.get('data', eval_data.get('instructions', []))
        is_dict = True
        outer_key = 'data' if 'data' in eval_data else 'instructions'
    else:
        items = eval_data
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
        new_item["translated_instruction"] = response
        updated_items.append(new_item)
        
        # 清理
        gc.collect()
        torch.cuda.empty_cache()

    # 保存结果
    output_data = {outer_key: updated_items} if is_dict else updated_items
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Done! Translated {len(updated_items)} samples")
    cleanup_temp_images()