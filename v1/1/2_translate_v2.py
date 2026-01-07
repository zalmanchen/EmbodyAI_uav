# 2_translate.py
import os
import json
import argparse
import torch
from PIL import Image
from io import BytesIO
import pandas as pd
from transformers import AutoProcessor, AutoModelForVision2Seq

# 配置
# from scripts.paths import DATASET_ROOT_PATH, OUTPUT_DIR, MODEL_ID

DATASET_ROOT_PATH = "../data/openfly/traj"
OUTPUT_DIR = "./train_t2rl_lora/data"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

TARGET_SIZE = (448, 448)

def load_images_from_parquet(item):
    """从 Parquet 加载图像 → List[bytes]"""
    parquet_path = os.path.join(DATASET_ROOT_PATH, f"{item['image_path']}.parquet")
    df = pd.read_parquet(parquet_path)
    images = []
    for idx_str in item['index_list']:
        idx = int(idx_str.split('_')[-1])
        img_bytes = df['image'][idx]['bytes']
        images.append(img_bytes)
    return images

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

def build_prompt(weaken_instr: str) -> str:
    return f"Human: {weaken_instr}\nAssistant: "

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "eval"], required=True)
    args = parser.parse_args()
    
    # 加载数据
    input_path = os.path.join(OUTPUT_DIR, f"t2rl_{args.split}.json")
    with open(input_path) as f:
        data = json.load(f)
    
    # 加载模型
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    # 翻译
    results = []
    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] {item.get('id', 'N/A')}")
        
        # 加载图像
        img_bytes_list = load_images_from_parquet(item)
        images = [
            Image.open(BytesIO(b)).convert("RGB").resize(TARGET_SIZE)
            for b in img_bytes_list
        ]
        
        # 构建 prompt
        # system = "You are a UAV instruction translator. Output precise English ONLY."
        # text = f"Human: {item['weaken_instruction']}\nAssistant: "
        text = build_prompt(item['weaken_instruction'])
        # 推理
        inputs = processor(
            text=text,
            images=images,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)
        response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 保存
        results.append({
            "id": item["id"],
            "weakened_instruction": item["weaken_instruction"],
            "translated_instruction": response.strip(),
            "image_bytes_list": img_bytes_list,  # 供 rollout 使用
            "trajectory_meta": {
                k: item[k] for k in ["image_path", "index_list", "pos", "yaw"]
                if k in item
            }
        })
    
    # 保存日志
    log_path = os.path.join(OUTPUT_DIR, f"logs_{args.split}_v0.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to {log_path}")

if __name__ == "__main__":
    main()