#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T²-RL Few-Shot 翻译器 | Qwen2.5-VL-7B + 8×A100 80G 专属优化
✅ 核心功能：
   - Few-Shot: --k_shot K (仅处理 K 条 train data)
   - 多样性采样: --n_samples N (每条指令 N 个翻译)
   - Few-Shot LoRA 训练支持: --disable_deterministic
✅ 无缝对接：
   - diverse_dpo_trainer.py
   - build_preference_pairs.py
   - 3_rollout.py
"""

import os
import json
import glob
import argparse
import gc
import torch
import random
import atexit
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq



# ========== 方案1补丁：安装qwen-vl-utils后，此行可注释 ==========
# from transformers.models.auto.auto_factory import _MODEL_MAPPING
# from transformers import Qwen2ForCausalLM
# _MODEL_MAPPING['qwen2_5_vl'] = Qwen2ForCausalLM

# ======================
# 🔧 全局配置【8×A100 80G + Qwen2.5-VL-7B 专属优化】
# ======================
DATASET_ROOT_PATH = "../data/openfly/traj"
OUTPUT_DIR = "./train_t2rl_lora/data"
MODEL_ID = "./model/qwen/Qwen2.5-VL-7B-Instruct"
TARGET_SIZE = (448, 448)
TEMP_IMAGE_DIR = "./tmp/qwen_vl_imgs"
CHUNK_SIZE = 16          # 80G显存专属，单批次16张图
MAX_NEW_TOKENS = 256
IMAGE_SAVE_QUALITY = 40

# ======================
# 🧼 初始化：临时目录+自动清理
# ======================
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cleanup_temp_images():
    for f in glob.glob(os.path.join(TEMP_IMAGE_DIR, "openfly_*.jpg")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"⚠️ [清理] 删除临时文件失败 {f}: {e}")

atexit.register(cleanup_temp_images)
print(f"✅ 自动清理已注册 | 临时图像目录: {TEMP_IMAGE_DIR}")

# ======================
# 🖼️ 图像预处理【8卡优化版】
# ======================
def save_image_to_temp(img_bytes: bytes, item_id: str, idx: int) -> str | None:
    try:
        fname = f"openfly_{item_id}_{idx:04d}_{os.getpid()}.jpg"
        temp_path = os.path.join(TEMP_IMAGE_DIR, fname)
        
        with BytesIO(img_bytes) as buf:
            img = Image.open(buf).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img.save(temp_path, "JPEG", quality=IMAGE_SAVE_QUALITY, optimize=True)
        
        with Image.open(temp_path) as saved_img:
            if saved_img.size != TARGET_SIZE or saved_img.mode != "RGB":
                raise ValueError(f"图像校验失败: {saved_img.size}/{saved_img.mode}")
        return temp_path
    except Exception as e:
        print(f"❌ [图像处理] 样本{item_id}帧{idx}失败: {e}")
        return None

def load_image_paths(item: dict) -> list:
    item_id = item.get("id", "unknown_id")
    parquet_name = item.get("image_path")
    index_list = item.get("index_list", [])
    
    if not parquet_name or not index_list:
        print(f"⚠️ [样本{item_id}] 缺失图像元信息")
        return []

    parquet_path = os.path.join(DATASET_ROOT_PATH, f"{parquet_name.strip()}.parquet")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ [样本{item_id}] Parquet加载失败: {e}")
        return []

    temp_img_paths = []
    for idx_str in index_list:
        try:
            frame_idx = int(str(idx_str).split('_')[-1])
            img_bytes = df["image"][frame_idx]["bytes"]
            temp_path = save_image_to_temp(img_bytes, item_id, frame_idx)
            if temp_path:
                temp_img_paths.append({"path": temp_path})
        except Exception as e:
            print(f"⚠️ [样本{item_id}] 跳过帧{idx_str}: {e}")
    
    print(f"✅ [样本{item_id}] 图像加载完成 | 原始{len(index_list)}帧 → 有效{len(temp_img_paths)}帧")
    return temp_img_paths

# ======================
# 📝 系统提示词【7B专属增强版】
# ======================
def build_system_prompt(agent_name: str = "openfly") -> str:
    role = (
        "You are an Intent-Driven Instruction Translator for UAV navigation.\n"
        "Your task is to convert vague human instructions into precise, executable commands "
        "that strictly follow the style of the target VLA agent. "
        "You will receive CONTINUOUS UAV trajectory frames (6~30+), follow the flight order strictly for translation."
    )
    few_shot = """
### Few-Shot Examples (OpenFly Style):
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
# 🤖 模型加载【Qwen2.5-VL-7B 完美适配版】
# ======================
def load_model_and_processor():
    print(f"\n🔍 加载模型 {MODEL_ID} | 适配8×A100 80G集群（7B专属优化）...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        resume_download=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced_low_0",
        load_in_4bit=False,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
    ).eval()
    
    print(f"✅ 模型加载完成 | GPU数量: {torch.cuda.device_count()} | 推理模式已启用")
    return processor, model

# ======================
# 🧠 单次推理【8卡极速版】
# ======================
def call_vlm_single(processor, model, images: list, instruction: str, system_prompt: str, 
                    disable_deterministic: bool = False) -> str:
    """确定性/非确定性推理"""
    valid_imgs = [img["path"] for img in images if os.path.exists(img["path"])]
    if not valid_imgs:
        return ""
    
    image_chunks = [valid_imgs[i:i + CHUNK_SIZE] for i in range(0, len(valid_imgs), CHUNK_SIZE)]
    all_outputs = []
    
    for chunk_paths in image_chunks:
        try:
            pil_images = [Image.open(p).convert("RGB").resize(TARGET_SIZE) for p in chunk_paths]
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_images],
                        {"type": "text", "text": instruction if len(all_outputs) == 0 else "Continue translating the trajectory strictly following the flight order."}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=pil_images,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(model.device)
            
            del pil_images
            gc.collect()
            
            # 配置生成参数
            if disable_deterministic:
                do_sample = True
                temperature = 0.1
                top_p = 0.95
                repetition_penalty = 1.1
            else:
                do_sample = False
                temperature = 0.0
                top_p = 1.0
                repetition_penalty = 1.0
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_cache=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    num_beams=1,
                )
            
            response = processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            all_outputs.append(response)
            
            del inputs, output_ids
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ [推理] 失败: {str(e)[:150]}")
            continue
    
    return " ".join(all_outputs).strip().replace("  ", " ").replace(". .", ".")

# ======================
# 🌈 多样性采样推理【Few-Shot LoRA 训练核心】
# ======================
def diverse_sample_vlm(processor, model, images: list, instruction: str, 
                       system_prompt: str, n_samples: int = 3) -> list[str]:
    """
    多样性采样：生成差异化翻译（用于 Diversity LoRA 训练）
    3 种策略覆盖多样性光谱：
      1. conservative: low temp, high top_p
      2. diverse: high temp, medium top_p
      3. anti-repeat: medium temp + rep penalty
    """
    if n_samples == 1:
        return [call_vlm_single(processor, model, images, instruction, system_prompt)]
    
    valid_imgs = [img["path"] for img in images if os.path.exists(img["path"])]
    if not valid_imgs:
        return [""]
    
    # 仅取第一块图像（避免长序列采样不稳定）
    pil_images = [Image.open(p).convert("RGB").resize(TARGET_SIZE) for p in valid_imgs[:CHUNK_SIZE]]
    messages = [{"role": "system", "content": system_prompt}]
    content = [{"type": "image", "image": img} for img in pil_images]
    content.append({"type": "text", "text": instruction})
    messages.append({"role": "user", "content": content})
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=pil_images,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = []
    strategies = [
        {"temp": 0.2, "top_p": 0.9, "rep_pen": 1.0, "name": "conservative"},
        {"temp": 0.7, "top_p": 0.8, "rep_pen": 1.0, "name": "diverse"},
        {"temp": 0.4, "top_p": 0.95, "rep_pen": 1.2, "name": "anti-repeat"}
    ]
    
    for i in range(min(n_samples, len(strategies))):
        cfg = strategies[i]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=cfg["temp"],
                top_p=cfg["top_p"],
                repetition_penalty=cfg["rep_pen"],
                use_cache=False,
            )
        response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        outputs.append(response)
        print(f"   🌈 Sample {i+1} ({cfg['name']}): {response[:80]}...")
    
    # 补齐样本（如果 n_samples > 3）
    while len(outputs) < n_samples:
        outputs.append(outputs[-1])  # 复用最后一个
    
    return outputs

# ======================
# 📊 GPU显存监控【8卡专属】
# ======================
def print_gpu_memory():
    if torch.cuda.is_available():
        total_used = 0.0
        print("   💾 8卡显存占用详情（80G）:")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            used = torch.cuda.memory_allocated() / 1024**3
            free = torch.cuda.mem_get_info()[0] / 1024**3
            total_used += used
            print(f"      GPU{i}: {used:.2f}GB / {80-used:.2f}GB free")

# ======================
# 🚀 主流程【Few-Shot 生产版】
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "eval"], required=True)
    parser.add_argument("--k_shot", type=int, default=None,
                        help="Few-shot: 随机采样 K 条数据（例如 5）")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="每条指令生成 N 个翻译（Few-Shot 建议 3）")
    parser.add_argument("--disable_deterministic", action="store_true",
                        help="Few-Shot 训练必需：禁用确定性推理")
    parser.add_argument("--seed", type=int, default=42,  # ✅ 新增
                        help="随机种子（Few-Shot 采样可复现）")
    args = parser.parse_args()
    
    # 加载数据
    input_json_path = os.path.join(OUTPUT_DIR, f"t2rl_{args.split}.json")
    if not os.path.exists(input_json_path):
        print(f"❌ 输入文件不存在: {input_json_path}")
        return
    
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # ✅ Few-Shot 核心：仅处理 K 条数据
    # original_count = len(data)
    # if args.k_shot is not None:
    #     data = data[:args.k_shot]
    #     print(f"🎯 Few-shot mode: K={args.k_shot} | 从 {original_count} 条中选取 {len(data)} 条")
    # else:
    #     print(f"📦 数据集加载完成 | 样本总数: {len(data)} | 划分类型: {args.split}")

    if args.k_shot is not None:
        # 设置随机种子确保可复现
        random.seed(args.seed)
        # 随机采样 K 条（无放回）
        if args.k_shot >= len(data):
            selected_data = data.copy()
            print(f"⚠️ K={args.k_shot} ≥ total {len(data)}, using all data")
        else:
            selected_data = random.sample(data, args.k_shot)
        data = selected_data
        print(f"🎯 Few-shot mode: K={args.k_shot} | 随机采样 {len(data)} 条 (seed={args.seed})")
        
    
    print(f"🚀 8×A100 80G集群已就绪 | n_samples={args.n_samples}")
    if args.disable_deterministic:
        print(f"   🔒 已禁用确定性推理（Few-Shot LoRA 训练模式）")
    
    # 初始化模型
    processor, model = load_model_and_processor()
    system_prompt = build_system_prompt()
    
    # 批量推理
    results = []
    success_count = 0
    
    for idx, item in enumerate(data):
        item_id = item.get("id", f"sample_{idx}")
        print(f"\n=================================================")
        print(f"--- [{idx+1}/{len(data)}] 处理样本: {item_id} ---")
        
        if idx % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        weaken_instr = item.get("weaken_instruction", "").strip()
        if not weaken_instr:
            results.append({"id":f"{item_id}_sample_1", "weakened_instruction":weaken_instr, "translated_instruction":"", "error":"无模糊指令"})
            continue
        
        image_paths = load_image_paths(item)
        if not image_paths:
            results.append({"id":f"{item_id}_sample_1", "weakened_instruction":weaken_instr, "translated_instruction":"", "error":"无有效轨迹图像"})
            continue
        
        # 生成翻译
        if args.disable_deterministic:
            # Few-Shot 训练模式：多样性采样
            translated_list = diverse_sample_vlm(
                processor, model, image_paths, weaken_instr, 
                system_prompt, 
                n_samples=args.n_samples
            )
        else:
            # 默认模式：确定性推理
            translated_list = [call_vlm_single(
                processor, model, image_paths, weaken_instr, 
                system_prompt, 
                disable_deterministic=False
            )]
        
        # 保存每个翻译为独立日志项
        for i, trans in enumerate(translated_list):
            strategy_name = "conservative"
            if args.disable_deterministic:
                strategy_name = ["conservative", "diverse", "anti-repeat"][min(i, 2)]
            
            res_item = {
                "id": f"{item_id}_sample_{i+1}",
                "weakened_instruction": weaken_instr,
                "translated_instruction": trans,
                "sample_id": i+1,
                "n_samples": args.n_samples,
                "k_shot": args.k_shot,
                "diversity_strategy": strategy_name,
                "disable_deterministic": args.disable_deterministic,
                "trajectory_meta": {
                    "image_path": item.get("image_path"),
                    "index_list": item.get("index_list"),
                    "pos": item.get("pos", []),
                    "yaw": item.get("yaw", [])
                }
            }
            results.append(res_item)
            if trans:
                success_count += 1
        
        print(f"✅ 样本{item_id} 完成 | 生成 {len(translated_list)} 个翻译")
        # print_gpu_memory()
    
    # 保存结果
    suffix_parts = []
    if args.k_shot is not None:
        suffix_parts.append(f"k{args.k_shot}")
    if args.disable_deterministic:
        suffix_parts.append("diverse")
    else:
        suffix_parts.append("deterministic")
    
    suffix = "_" + "_".join(suffix_parts)
    log_save_path = os.path.join(OUTPUT_DIR, f"logs_{args.split}_8xA100_7B_n{args.n_samples}{suffix}.json")
    with open(log_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 构建训练数据（合并到原数据）
    sample_map = {}
    for res in results:
        base_id = res["id"].split("_sample_")[0]
        if base_id not in sample_map:
            sample_map[base_id] = []
        sample_map[base_id].append(res)
    
    new_data = []
    for item in data:
        base_id = item.get("id", "unknown")
        if base_id in sample_map and sample_map[base_id]:
            # 取第一个翻译作为默认
            item["translated_instruction"] = sample_map[base_id][0]["translated_instruction"]
        else:
            item["translated_instruction"] = ""
        new_data.append(item)
    
    raw_save_path = os.path.join(OUTPUT_DIR, f"t2rl_{args.split}_with_translated.json")
    with open(raw_save_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    # 输出统计
    total_trans = len(data) * args.n_samples
    success_rate = (success_count / total_trans) * 100
    print(f"\n=================================================")
    print(f"✅ ✅ T²-RL Few-Shot 翻译完成！")
    print(f"📊 样本数: {len(data)} (K={args.k_shot}) | 总翻译数: {total_trans} | 成功率: {success_rate:.2f}%")
    print(f"📁 详细日志: {log_save_path}")
    print(f"📁 训练数据: {raw_save_path}")

# ======================
# 🔍 程序入口
# ======================
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["TORCH_USE_CUDA_DSA"] = "0"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    torch.cuda.empty_cache()
    gc.collect()
    main()