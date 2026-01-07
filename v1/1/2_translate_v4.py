# 2_translate.py
import os
import json
import argparse
import gc
import torch
from PIL import Image
from io import BytesIO
import pandas as pd
from transformers import AutoProcessor, AutoModelForVision2Seq

# 配置
DATASET_ROOT_PATH = "../data/openfly/traj"
OUTPUT_DIR = "./train_t2rl_lora/data"
MODEL_ID = "./model/Qwen/Qwen3-VL-8B-Instruct"
TARGET_SIZE = (448, 448)

def load_images_from_parquet(item):
    """从 Parquet 加载图像 → List[PIL.Image] (448x448 RGB)"""
    parquet_path = os.path.join(DATASET_ROOT_PATH, f"{item['image_path']}.parquet")
    df = pd.read_parquet(parquet_path)
    images = []
    for idx_str in item['index_list']:
        idx = int(idx_str.split('_')[-1])
        img_bytes = df['image'][idx]['bytes']
        # 直接返回 PIL Image（避免中间 bytes 列表占内存）
        with BytesIO(img_bytes) as buf:
            img = Image.open(buf).convert("RGB").resize(TARGET_SIZE)
            images.append(img.copy())  # .copy() 防文件句柄
    return images

def build_system_prompt(agent_name: str = "openfly") -> str:
    role = (
        "You are an Intent-Driven Instruction Translator for UAV navigation.\n"
        "Your task is to convert vague human instructions into precise, executable commands "
        "that strictly follow the style of the target VLA agent."
    )
    few_shot = """
### Few-Shot Examples (OpenFly Style):
"Head directly toward the tall , light beige building with many windows . Then , slightly turn right and proceed to another large building characterized by its light gray color and balcony - like structures . Finish by slightly turning left , continuing straight towards a tall , multi - story skyscraper with large , beige windows featuring arched tops ."
"Proceed directly to the grey urban rooftop featuring antennas and equipment on a medium - sized building , then slightly turn left and head straight towards it ."
"Advance towards the gray skyscraper characterized by a tall building , then slightly turn right slightly and proceed to it . Finally , slightly turn left and continue straight to it ."
""".strip()
    constraints = (
        "\n### Critical Constraints:\n"
        "- STRICTLY ground all descriptions in the provided images.\n"
        "- PRESERVE the user's core intent.\n"
        "- DO NOT hallucinate objects absent in visuals.\n"
        "- Output fluent, multi-step, descriptive English."
    )
    return f"{role}\n\n{few_shot}\n{constraints}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "eval"], required=True)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device (8xA100: 4-8)")  # ✅ 新增
    args = parser.parse_args()
    
    # 加载数据
    input_path = os.path.join(OUTPUT_DIR, f"t2rl_{args.split}.json")
    with open(input_path) as f:
        data = json.load(f)
    print(f"📦 Loaded {len(data)} items. Using batch_size={args.batch_size}")

    # 加载模型（8xA100 全利用）
    print("🔍 Loading Qwen3-VL-8B-Instruct...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # ✅ 自动分配到 8 卡
        # device_map="balanced",  # 或显式平衡
    ).eval()
    print(f"✅ Model loaded on {torch.cuda.device_count()} GPUs")

    # 系统提示
    system_prompt = build_system_prompt()
    
    results = []
    
    # ✅ 批量处理（充分利用多卡）
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i+args.batch_size]
        print(f"\n--- Batch {i//args.batch_size + 1}/{(len(data)-1)//args.batch_size + 1} ({len(batch)} items) ---")
        
        batch_inputs = []
        batch_items = []
        
        # 构建 batch 输入
        for item in batch:
            try:
                print(f"  📥 Loading images for {item.get('id', 'N/A')}...")
                images = load_images_from_parquet(item)
                
                # ✅ 关键：使用 messages 格式（支持 system）
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            *[
                                {"type": "image", "image": img} 
                                for img in images
                            ],
                            {"type": "text", "text": item['weaken_instruction']}
                        ]
                    }
                ]
                
                # Tokenize
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                )
                inputs = processor(
                    text=text,       # List[str] 
                    images=images,   # List[List[PIL.Image]]
                    padding=True,    # ✅ 启用 padding
                    trunction=True,  # ✅ 启用 truncation
                    return_tensors="pt"
                )
                
                batch_inputs.append(inputs)
                batch_items.append((item, images))  # 保存 images 用于后续清理
                
            except Exception as e:
                print(f"  ❌ Preprocess failed for {item.get('id')}: {e}")
                results.append({
                    "id": item.get("id"),
                    "weakened_instruction": item.get("weaken_instruction", ""),
                    "translated_instruction": "",
                    "translation_error": f"Preprocess: {str(e)}"
                })
                continue
        
        if not batch_inputs:
            continue
        
        # 合并 batch（pad）
        try:
            from transformers import DataCollatorForSeq2Seq
            collator = DataCollatorForSeq2Seq(
                tokenizer=processor.tokenizer,
                padding=True,
                return_tensors="pt"
            )
            batch_dict = collator(batch_inputs)
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
        except Exception as e:
            print(f"  ❌ Batch collate failed: {e}")
            # 退化为单条处理
            for (item, images), inputs in zip(batch_items, batch_inputs):
                try:
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            use_cache=False,  # ✅ 减少显存
                        )
                    response = processor.decode(
                        output_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    results.append({
                        "id": item["id"],
                        "weakened_instruction": item["weaken_instruction"],
                        "translated_instruction": response,
                        "trajectory_meta": {k: item[k] for k in ["image_path", "index_list"] if k in item}
                    })
                except Exception as e2:
                    results.append({
                        "id": item["id"],
                        "weakened_instruction": item["weaken_instruction"],
                        "translated_instruction": "",
                        "translation_error": f"Generate: {str(e2)}"
                    })
                # ✅ 清理资源
                del inputs
                for img in images:
                    img.close()
                gc.collect()
                torch.cuda.empty_cache()
            continue
        
        # 批量推理
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **batch_dict,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=False,  # ✅ 关键：8B 模型必须关 KV Cache
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # 解码
            for j, (item, images) in enumerate(batch_items):
                try:
                    response = processor.decode(
                        output_ids[j][batch_dict["input_ids"][j].shape[0]:],
                        skip_special_tokens=True
                    ).strip()
                    results.append({
                        "id": item["id"],
                        "weakened_instruction": item["weaken_instruction"],
                        "translated_instruction": response,
                        "trajectory_meta": {
                            k: item[k] for k in ["image_path", "index_list", "pos", "yaw"]
                            if k in item
                        }
                    })
                except Exception as e:
                    results.append({
                        "id": item["id"],
                        "weakened_instruction": item["weaken_instruction"],
                        "translated_instruction": "",
                        "translation_error": f"Decode: {str(e)}"
                    })
                
                # ✅ 清理 PIL 图像
                for img in images:
                    img.close()
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ❌ CUDA OOM in batch: {e}. Reducing batch_size...")
            # 退化为单条处理（同上）
            for (item, images), inputs in zip(batch_items, batch_inputs):
                # ... 单条处理逻辑（同上） ...
                pass
        
        # ✅ 关键：清理 batch 显存
        del batch_dict, output_ids
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  🧹 CUDA cache cleared. Free GPU: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")

    # 保存结果
    log_path = os.path.join(OUTPUT_DIR, f"logs_{args.split}_v1.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ All done! Saved {len(results)}/{len(data)} items to {log_path}")
    
    # 统计
    success = sum(1 for r in results if r.get("translated_instruction"))
    print(f"📊 Success: {success}/{len(data)} ({success/len(data)*100:.1f}%)")

if __name__ == "__main__":
    main()