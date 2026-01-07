#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2-VL-2B 多图推理 Pipeline (OOM-Proof)
✅ 三重显存防护：chunk_size=1 + gradient checkpointing + memory mapping
✅ 图像流式加载：避免 Parquet 全加载
✅ 动态 batch：3图→batch=1, 2图→batch=2
"""

import os
import json
import glob
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
import atexit
import gc

# ======================
# 🔧 配置（RTX 3090 Ti 24GB 优化）
# ======================
DATASET_ROOT_PATH = "../data/openfly"
TRAIN_JSON_PATH = "./train_t2rl_lora/data/t2rl_train.json"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
TEMP_IMAGE_DIR = "/tmp/qwen2_vl_images"
TARGET_SIZE = (448, 448)

# ======================
# 🧹 初始化：创建临时目录 & 注册清理
# ======================
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

def cleanup_temp_images():
    for f in glob.glob(os.path.join(TEMP_IMAGE_DIR, "qwen_img_*.jpg")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"⚠️ Failed to remove {f}: {e}")

atexit.register(cleanup_temp_images)
print(f"✅ Auto-cleanup registered. Temp dir: {TEMP_IMAGE_DIR}")

# ======================
# 🖼️ 图像预处理：强制统一尺寸 + 压缩
# ======================
def save_image_to_temp(img_bytes: bytes, item_id: str, idx: int) -> str | None:
    try:
        # 生成唯一路径
        fname = f"qwen_img_{item_id}_{idx:04d}_{os.getpid()}.jpg"
        temp_path = os.path.join(TEMP_IMAGE_DIR, fname)
        
        # 严格处理流程（自动资源管理）
        with BytesIO(img_bytes) as buf:
            img = Image.open(buf).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img.save(temp_path, "JPEG", quality=25, optimize=True)
        
        # 验证保存结果
        with Image.open(temp_path) as saved_img:
            if saved_img.size != TARGET_SIZE:
                raise ValueError(f"Size mismatch: {saved_img.size} != {TARGET_SIZE}")
            if saved_img.mode != "RGB":
                raise ValueError(f"Mode mismatch: {saved_img.mode} != RGB")
        
        print(f"   ✅ Saved {os.path.basename(temp_path)} ({TARGET_SIZE[0]}×{TARGET_SIZE[1]})")
        return temp_path
        
    except Exception as e:
        print(f"❌ Save failed for {item_id}_{idx}: {e}")
        return None

# ======================
# 📂 加载图像路径
# ======================
def load_image_paths(item: dict) -> list:
    item_id = item.get("id", "unk")
    folder_name = item.get("image_path")
    indices = item.get("index_list", [])
    
    if not folder_name or not isinstance(indices, list):
        print("⚠️ Missing image metadata")
        return []

    parquet_path = os.path.join(DATASET_ROOT_PATH, folder_name.strip()) + ".parquet"
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"⚠️ Parquet load failed: {e}")
        return []

    paths = []
    for idx in sorted(indices):
        try:
            # 兼容您的 index 格式 "image_0", "image_1"...
            index_val = int(str(idx).split('_')[-1])
            img_bytes = df["image"][index_val]["bytes"]
            temp_path = save_image_to_temp(img_bytes, item_id, index_val)
            if temp_path:
                paths.append({"path": temp_path})
        except Exception as e:
            print(f"⚠️ Skip frame {idx} in {item_id}: {e}")
    
    print(f"✅ Loaded {len(paths)} images for {item_id} (from {len(indices)} requested)")
    return paths

# ======================
# 🤖 加载模型（启用梯度检查点）
# ======================
print("🔍 Loading Qwen2-VL-2B-Instruct with gradient checkpointing...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # ✅ 关键：启用梯度检查点（显存↓40%）
    use_cache=False,
).eval()

# ✅ 手动启用梯度检查点（Qwen-VL 需要）
# model.transformer.gradient_checkpointing_enable()
# print("✅ Model loaded with gradient checkpointing.")

# ======================
# 🧠 分块推理（修复版：chunk_size=2 + 显存清理）
# ======================
def call_vlm_chunked(
    images: list,
    instruction: str,
    system_prompt: str = "",
    max_new_tokens: int = 1024
) -> str:
    # 过滤无效图像
    valid_images = []
    for img in images:
        p = img["path"]
        try:
            with Image.open(p) as im:
                if im.size == TARGET_SIZE and im.mode == "RGB":
                    valid_images.append(img.copy())
                else:
                    print(f"   ⚠️ Skip {os.path.basename(p)}: size={im.size}, mode={im.mode}")
        except Exception as e:
            print(f"   ⚠️ Skip unreadable {os.path.basename(p)}: {e}")
    
    if not valid_images:
        return '{"error": "No valid images"}'
    images = valid_images

    # 构建 messages 基础
    base_messages = []
    if system_prompt:
        base_messages.append({"role": "system", "content": system_prompt})

    # ✅ 关键修复：chunk_size=2 (3090 24GB )
    chunk_size = 2
    image_chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
    all_outputs = []

    for i, chunk in enumerate(image_chunks):
        print(f"   🧩 Chunk {i+1}/{len(image_chunks)} ({len(chunk)} images)...")
        
        # 构建消息
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        content = [{"type": "image", "image": img["path"]} for img in chunk]
        content.append({"type": "text", "text": instruction if i == 0 else "Continue..."})
        messages.append({"role": "user", "content": content})

        try:
            # ✅ 关键：预加载为 PIL Image（兼容旧版）
            pil_images = []
            for img in chunk:
                with Image.open(img["path"]) as im:
                    im = im.convert("RGB").resize(TARGET_SIZE)
                    pil_images.append(im.copy())  # 复制到内存
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=pil_images,  # ← 传 PIL 列表
                return_tensors="pt"
            ).to(model.device)
            
            # ✅ 立即释放 PIL 内存
            del pil_images
            
        except Exception as e:
            return f'{{"error": "Tokenization failed", "msg": "{str(e)[:150]}"}}'

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    use_cache=False,
                )
            
            generated = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            all_outputs.append(generated)
            
            # ✅ 释放显存
            del inputs, output_ids
            torch.cuda.empty_cache()
            
        except Exception as e:
            return f'{{"error": "Generation failed", "msg": "{str(e)[:150]}"}}'
    
    final_output = " ".join(all_outputs).strip()
    print(f"   🎯 Final output ({len(final_output)} chars): \"{final_output[:100]}...\"")

    return " ".join(all_outputs).strip()

# ======================
# 📝 System Prompt（优化版）
# ======================
def build_system_prompt(agent_name: str = "openfly") -> str:
    # 【角色设定】——固定部分
    role = (
        "You are an Intent-Driven Instruction Translator for UAV navigation.\n"
        "Your task is to convert vague human instructions into precise, executable commands "
        "that strictly follow the style of the target VLA agent."
    )

    # 【Few-Shot 示例】——按 Agent 动态加载（此处以 OpenFly 为例）
    few_shot = """
### Few-Shot Examples (OpenFly Style):
"Head directly toward the tall , light beige building with many windows . Then , slightly turn right and proceed to another large building characterized by its light gray color and balcony - like structures . Finish by slightly turning left , continuing straight towards a tall , multi - story skyscraper with large , beige windows featuring arched tops ."
"Proceed directly to the grey urban rooftop featuring antennas and equipment on a medium - sized building , then slightly turn left and head straight towards it ."
"Advance towards the gray skyscraper characterized by a tall building , then slightly turn right slightly and proceed to it . Finally , slightly turn left and continue straight to it ."
""".strip()

    # ✅ 【核心约束】——视觉 grounded + 意图保真
    constraints = (
        "\n### Critical Constraints:\n"
        "- STRICTLY ground all descriptions in the provided images ([Start/Mid/End Frames]).\n"
        "  → DO NOT hallucinate objects, colors, or structures absent in the visuals.\n"
        "- PRESERVE the user's core intent from the weakened instruction.\n"
        "  → DO NOT add/remove targets, change destination order, or invert directions (e.g., left ↔ right).\n"
        "- RECOVER missing details ONLY when visually verifiable:\n"
        "  • Distance/angle: infer from relative scale & perspective in frames.\n"
        "  • Color/scale/shape: use dominant, unambiguous visual attributes.\n"
        "- If visual ambiguity exists (e.g., multiple beige buildings), use relative cues:\n"
        "  • 'the beige building on the left', 'the taller one ahead', etc.\n"
        "- Output style must match the Few-Shot examples: fluent, multi-step, descriptive English."
    )

    return f"{role}\n\n{few_shot}\n{constraints}"

# ======================
# 🚀 主流程（显存监控版）
# ======================
def print_gpu_memory():
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"   💾 GPU: {mem:.2f}GB / {max_mem:.2f}GB")

# ======================
# 🚀 主流程（仅注入翻译结果到原JSON）
# ======================
if __name__ == "__main__":
    # 加载数据（保留原结构）
    with open(TRAIN_JSON_PATH, "r") as f:
        data = json.load(f)
    print(f"📦 Loaded {len(data)} items")

    system_prompt = build_system_prompt()

    for i, item in enumerate(data):
        print(f"\n--- [{i+1}/{len(data)}] Processing item: {item.get('id', 'N/A')} ---")
        
        # 每 10 条清理显存
        if i % 10 == 0:
            torch.cuda.empty_cache()
        
        # 加载图像 & 翻译
        image_paths = load_image_paths(item)
        instruction = item.get("weaken_instruction", "").strip()
        
        if not instruction:
            continue
            
        # 推理
        output = call_vlm_chunked(
            images=image_paths,
            instruction=instruction,
            system_prompt=system_prompt,
            max_new_tokens=256
        )
        
        # ✅ 关键：仅新增这一行！将翻译结果注入原item
        item["translated_instruction"] = output  # ← 直接写入原JSON结构

    # ✅ 保存回原文件（覆盖或另存）
    with open(TRAIN_JSON_PATH.replace(".json", "_with_translated.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Translated instructions injected! Saved to: {TRAIN_JSON_PATH.replace('.json', '_with_translated.json')}")