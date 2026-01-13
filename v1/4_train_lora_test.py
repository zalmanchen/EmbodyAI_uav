#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal DPO Training + Inference Script
✅ Complete pipeline: Train → Infer → Save Results
"""

import os
import json
import torch
import argparse
import warnings
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from transformers import (
    TrainingArguments, 
    AutoModelForVision2Seq, 
    AutoProcessor,
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel

# 配置路径
DATASET_ROOT_PATH = "../data/openfly/traj"
TEMP_IMAGE_DIR = './tmp/qwen_vl_imgs'
TARGET_SIZE = (448, 448)

# ======================
# 📊 数据集构建
# ======================

def build_multimodal_dpo_dataset(rollout_path: str) -> Dataset:
    """构建多模态 DPO 数据集 - 训练用"""
    with open(rollout_path, 'r') as f:
        logs = json.load(f)

    dpo_list = []
    
    for log_idx, log in enumerate(logs):
        print(f"🔍 Processing sample {log_idx}: {log.get('image_path', 'N/A')}")
        
        # 提取基本信息
        image_path = log.get("image_path", "")
        index_list = log.get("index_list", [])
        weaken_instruction = log.get("weaken_instruction", "").strip()
        weaken_reward = log.get("weaken_instruction_reward") 
        
        if not weaken_instruction or not index_list:
            print(f"  ⚠️ Skip sample {log_idx}: missing instruction or frames")
            continue
        
        # 提取 3 种翻译及其奖励
        candidates = []
        for i in range(1, 4):
            text_key = f"translated_instruction_{i}"
            reward_key = f"translated_instruction_{i}_reward"
            
            if text_key in log and reward_key in log:
                text = log[text_key].strip()
                try:
                    reward = float(log[reward_key])
                    if text and len(text) > 10:
                        candidates.append({
                            "text": text,
                            "reward": reward,
                            "id": i
                        })
                except (ValueError, TypeError) as e:
                    print(f"  ⚠️ Sample {log_idx} translation {i} reward error: {e}")
        
        print(f"  ✅ {len(candidates)} valid translations found")        
        if len(candidates) < 2:
            continue
        
        # 按 reward 排序
        candidates.sort(key=lambda x: x["reward"], reverse=True)

        # 🔑 关键条件：最高奖励必须高于 weaken_reward
        max_reward = candidates[0]["reward"]
        if max_reward <= weaken_reward:
            print(f"  ⚠️ Skip sample {log_idx}: max_reward ({max_reward:.3f}) <= weaken_reward ({weaken_reward:.3f})")
            continue
        
        print(f"  ✅ Accepted: max_reward ({max_reward:.3f}) > weaken_reward ({weaken_reward:.3f})")
        
        # 创建图像路径
        image_paths = []
        _temp_path = os.path.join(TEMP_IMAGE_DIR, image_path)
        os.makedirs(_temp_path, exist_ok=True)

        parquet_path = os.path.join(DATASET_ROOT_PATH, f"{image_path.strip()}.parquet")
        try:
            df = pd.read_parquet(parquet_path)
        except:
            return []

        for index in index_list:
            fname = index + ".jpg"
            tmp_path = os.path.join(_temp_path, fname)
            image_paths.append(tmp_path)

            if os.path.exists(tmp_path):
                continue

            frame_idx = int(str(index).split('_')[-1])
            img_bytes = df["image"][frame_idx]["bytes"]

            with BytesIO(img_bytes) as buf:
                img = Image.open(buf).convert("RGB")
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                img.save(tmp_path, "JPEG", quality=40)

        # 构建 DPO 样本对
        if len(candidates) >= 2:
            dpo_list.append({
                "images": image_paths,
                "prompt": weaken_instruction,
                "chosen": candidates[0]["text"],
                "rejected": candidates[-1]["text"],
                "reward_gap": candidates[0]["reward"] - candidates[-1]["reward"],
                "pair_type": "max-min"
            })
        
        if len(candidates) >= 3:
            dpo_list.append({
                "images": image_paths,
                "prompt": weaken_instruction,
                "chosen": candidates[0]["text"],
                "rejected": candidates[1]["text"],
                "reward_gap": candidates[0]["reward"] - candidates[1]["reward"],
                "pair_type": "max-second"
            })
        
        if len(candidates) >= 3:
            dpo_list.append({
                "images": image_paths,
                "prompt": weaken_instruction,
                "chosen": candidates[1]["text"],
                "rejected": candidates[-1]["text"],
                "reward_gap": candidates[1]["reward"] - candidates[-1]["reward"],
                "pair_type": "second-min"
            })
    
    if not dpo_list:
        raise ValueError("❌ No valid DPO pairs generated!")
    
    dpo_dataset = Dataset.from_list(dpo_list)
    print(f"\n✅ Multimodal DPO dataset built: {len(dpo_dataset)} pairs")
    return dpo_dataset

def load_inference_dataset(rollout_path: str, max_samples: int = None):
    """加载推理数据集"""
    with open(rollout_path, 'r') as f:
        logs = json.load(f)
    
    inference_data = []
    for log_idx, log in enumerate(logs):
        if max_samples and len(inference_data) >= max_samples:
            break
            
        image_path = log.get("image_path", "")
        index_list = log.get("index_list", [])
        weaken_instruction = log.get("weaken_instruction", "").strip()
        
        if not weaken_instruction or not index_list:
            continue
        
        # 创建临时图像路径
        image_paths = []
        _temp_path = os.path.join(TEMP_IMAGE_DIR, image_path)
        os.makedirs(_temp_path, exist_ok=True)
        
        parquet_path = os.path.join(DATASET_ROOT_PATH, f"{image_path.strip()}.parquet")
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"⚠️ Parquet loading failed for {image_path}: {e}")
            continue
        
        for index in index_list:
            fname = index + ".jpg"
            tmp_path = os.path.join(_temp_path, fname)
            image_paths.append(tmp_path)
            
            if os.path.exists(tmp_path):
                continue
                
            try:
                frame_idx = int(str(index).split('_')[-1])
                img_bytes = df["image"][frame_idx]["bytes"]
                with BytesIO(img_bytes) as buf:
                    img = Image.open(buf).convert("RGB")
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    img.save(tmp_path, "JPEG", quality=40)
            except Exception as e:
                print(f"⚠️ Image processing failed for {index}: {e}")
                continue
        
        inference_data.append({
            "original_index": log_idx,
            "image_path": image_path,
            "images": image_paths,
            "prompt": weaken_instruction,
            "original_translations": {
                f"translated_{i}": log.get(f"translated_instruction_{i}", ""),
                f"reward_{i}": log.get(f"translated_instruction_{i}_reward", 0.0)
                for i in range(1, 4)
            },
            "weaken_reward": log.get("weaken_instruction_reward", 0.0)
        })
    
    print(f"✅ Loaded {len(inference_data)} samples for inference")
    return inference_data

# ======================
# 🔧 Collator 和 Trainer
# ======================

class MultimodalDPOCollator:
    """多模态 DPO collator"""
    
    def __init__(self, processor, max_frames=8, system_prompt=None):
        self.processor = processor
        self.max_frames = max_frames
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.pad_token_id = 151643
        self.processor.tokenizer.eos_token_id = 151645
    
    def _default_system_prompt(self):
        return (
            "You are a precision UAV instruction translator specialized in aerial navigation.\n"
            "Your task is to convert high-level human instructions into detailed, executable flight commands\n"
            "that maintain strict adherence to the observed visual trajectory and flight sequence."
        )
    
    def __call__(self, features):
        from qwen_vl_utils import process_vision_info
        
        prompt_messages = []
        chosen_responses = []
        rejected_responses = []
        all_vision_inputs = []
        
        for item in features:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": []}
            ]
            
            image_paths = item["images"][:self.max_frames]
            for img_path in image_paths:
                if isinstance(img_path, dict):
                    path_str = img_path.get("path", "")
                else:
                    path_str = img_path
                
                if path_str:
                    if os.path.exists(path_str.replace("file://", "")):
                        abs_path = os.path.abspath(path_str.replace("file://", ""))
                        messages[1]["content"].append({"type": "image", "image": f"file://{abs_path}"})
                    else:
                        messages[1]["content"].append({"type": "image", "image": path_str})
            
            messages[1]["content"].append({"type": "text", "text": item["prompt"]})
            prompt_messages.append(messages)
            chosen_responses.append(item["chosen"])
            rejected_responses.append(item["rejected"])
            
            try:
                vision_input, _ = process_vision_info(messages)
                if vision_input:
                    all_vision_inputs.extend(vision_input)
            except Exception as e:
                print(f"⚠️ Vision processing failed: {e}")
                continue
        
        prompt_texts = []
        for messages in prompt_messages:
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt_texts.append(text)
            except Exception as e:
                print(f"⚠️ Chat template failed: {e}")
                num_imgs = len([c for msg in messages for c in (msg.get("content", []) if isinstance(msg.get("content"), list) else [msg.get("content")]) if isinstance(c, dict) and c.get("type") == "image"])
                placeholder = "<tool_call>" * num_imgs + "<tool_call>"
                system_part = f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
                user_part = f"<|im_start|>user\n{placeholder}\n{messages[-1]['content'][-1]['text']}<|im_end|>"
                prompt_texts.append(f"{system_part}\n{user_part}")
        
        if all_vision_inputs:
            prompt_inputs = self.processor(
                text=prompt_texts,
                images=all_vision_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            prompt_inputs = self.processor.tokenizer(
                prompt_texts,
                padding=True,
                return_tensors="pt"
            )
        
        chosen_inputs = self.processor.tokenizer(
            chosen_responses,
            padding=True,
            return_tensors="pt"
        )
        rejected_inputs = self.processor.tokenizer(
            rejected_responses,
            padding=True,
            return_tensors="pt"
        )
        
        chosen_labels = chosen_inputs.input_ids.clone()
        chosen_labels[chosen_labels == self.processor.tokenizer.pad_token_id] = -100
        rejected_labels = rejected_inputs.input_ids.clone()
        rejected_labels[rejected_labels == self.processor.tokenizer.pad_token_id] = -100
        
        result = {
            "prompt_input_ids": prompt_inputs.input_ids,
            "prompt_attention_mask": prompt_inputs.attention_mask,
            "chosen_input_ids": chosen_inputs.input_ids,
            "chosen_attention_mask": chosen_inputs.attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_inputs.input_ids,
            "rejected_attention_mask": rejected_inputs.attention_mask,
            "rejected_labels": rejected_labels,
        }
        
        vision_keys = ['pixel_values', 'image_grid_thw']
        for key in vision_keys:
            if hasattr(prompt_inputs, key) and getattr(prompt_inputs, key) is not None:
                result[key] = getattr(prompt_inputs, key)
                
        return result

class MultimodalDPOTrainer(Trainer):
    """多模态 DPO Trainer"""
    
    def __init__(self, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
        
        # 拼接 prompt + chosen/rejected
        full_chosen_input_ids = torch.cat([
            filtered_inputs["prompt_input_ids"],
            filtered_inputs["chosen_input_ids"]
        ], dim=1)
        
        full_chosen_attention_mask = torch.cat([
            filtered_inputs["prompt_attention_mask"],
            filtered_inputs["chosen_attention_mask"]
        ], dim=1)
        
        full_chosen_labels = torch.cat([
            torch.full_like(filtered_inputs["prompt_input_ids"], -100),
            filtered_inputs["chosen_labels"]
        ], dim=1)
        
        full_rejected_input_ids = torch.cat([
            filtered_inputs["prompt_input_ids"],
            filtered_inputs["rejected_input_ids"]
        ], dim=1)
        
        full_rejected_attention_mask = torch.cat([
            filtered_inputs["prompt_attention_mask"],
            filtered_inputs["rejected_attention_mask"]
        ], dim=1)
        
        full_rejected_labels = torch.cat([
            torch.full_like(filtered_inputs["prompt_input_ids"], -100),
            filtered_inputs["rejected_labels"]
        ], dim=1)
        
        vision_kwargs = {}
        if "pixel_values" in filtered_inputs:
            vision_kwargs["pixel_values"] = filtered_inputs["pixel_values"]
        if "image_grid_thw" in filtered_inputs:
            vision_kwargs["image_grid_thw"] = filtered_inputs["image_grid_thw"]
        
        chosen_outputs = model(
            input_ids=full_chosen_input_ids,
            attention_mask=full_chosen_attention_mask,
            labels=full_chosen_labels,
            **vision_kwargs
        )
        
        rejected_outputs = model(
            input_ids=full_rejected_input_ids,
            attention_mask=full_rejected_attention_mask,
            labels=full_rejected_labels,
            **vision_kwargs
        )
        
        chosen_logps = -chosen_outputs.loss
        rejected_logps = -rejected_outputs.loss
        logits = self.beta * (chosen_logps - rejected_logps)
        dpo_loss = -torch.nn.functional.logsigmoid(logits).mean()
        
        return dpo_loss

# ======================
# 🔧 LoRA 配置
# ======================
def configure_lora(model, target_modules=None):
    """配置 LoRA"""
    if target_modules is None:
        target_modules = []
        try:
            num_layers = len(model.language_model.layers)
            for i in range(num_layers):
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    module_name = f"language_model.layers.{i}.self_attn.{proj}"
                    try:
                        _ = model.get_submodule(module_name)
                        target_modules.append(module_name)
                    except AttributeError:
                        continue
        except Exception as e:
            print(f"⚠️ Auto-detection failed: {e}")
            target_modules = [
                r"language_model\.layers\..*\.self_attn\.q_proj",
                r"language_model\.layers\..*\.self_attn\.k_proj",
                r"language_model\.layers\..*\.self_attn\.v_proj",
                r"language_model\.layers\..*\.self_attn\.o_proj",
            ]
    
    print(f"🔧 Configuring LoRA with {len(target_modules)} target modules")
    
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.15,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model

# ======================
# 🚀 训练函数
# ======================
def run_training(
    exp_name: str,
    rollout_path: str,
    output_dir: str,
    epochs: int = 3,
    **kwargs
):
    """运行训练"""
    print(f"\n🚀 Starting training: {exp_name}")
    
    # 1. 构建数据集
    dpo_dataset = build_multimodal_dpo_dataset(rollout_path)
    
    # 2. 加载模型
    print("🔍 Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "./model/qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 3. 配置 LoRA
    peft_model = configure_lora(model)
    
    # 4. 处理器
    print("🔍 Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "./model/qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.bos_token_id = 151644
    
    # 5. 计算保存步数（每5个epochs）
    steps_per_epoch = len(dpo_dataset) // (1 * 1)  # batch_size=1, grad_acc=1
    if len(dpo_dataset) % (1 * 1) != 0:
        steps_per_epoch += 1
    save_steps = steps_per_epoch * 5
    
    # 6. 训练配置
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, exp_name),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="no",
        include_inputs_for_metrics=False,
        prediction_loss_only=True,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
    )
    
    # 7. 创建 trainer
    collator = MultimodalDPOCollator(processor, max_frames=60)
    
    trainer = MultimodalDPOTrainer(
        model=peft_model,
        args=training_args,
        beta=0.1,
        train_dataset=dpo_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    
    # 8. 开始训练
    print(f"🎯 Training {exp_name} with {len(dpo_dataset)} samples...")
    trainer.train()
    
    # 9. 保存模型
    final_path = os.path.join(output_dir, exp_name, "final")
    peft_model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"✅ Training completed! Model saved to: {final_path}")
    
    return final_path

# ======================
# 🎯 推理函数
# ======================
def build_system_prompt():
    """构建系统提示"""
    return (
        "You are a precision UAV instruction translator specialized in aerial navigation.\n"
        "Your task is to convert high-level human instructions into detailed, executable flight commands\n"
        "that maintain strict adherence to the observed visual trajectory and flight sequence."
    )

def run_inference(model, processor, inference_data, max_new_tokens=512, temperature=0.7):
    """运行推理"""
    results = []
    system_prompt = build_system_prompt()
    
    for sample in tqdm(inference_data, desc="Running inference"):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": []}
            ]
            
            for img_path in sample["images"]:
                if os.path.exists(img_path):
                    messages[1]["content"].append({
                        "type": "image", 
                        "image": f"file://{os.path.abspath(img_path)}"
                    })
            
            messages[1]["content"].append({"type": "text", "text": sample["prompt"]})
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            if "assistant" in generated_text:
                response = generated_text.split("assistant")[-1].strip()
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()
            else:
                response = generated_text.strip()
            
            results.append({
                "original_index": sample["original_index"],
                "image_path": sample["image_path"],
                "prompt": sample["prompt"],
                "generated_response": response,
                "original_translations": sample["original_translations"],
                "weaken_reward": sample["weaken_reward"]
            })
            
        except Exception as e:
            print(f"❌ Inference failed for sample {sample['original_index']}: {e}")
            results.append({
                "original_index": sample["original_index"],
                "image_path": sample["image_path"],
                "prompt": sample["prompt"],
                "generated_response": f"ERROR: {str(e)}",
                "original_translations": sample["original_translations"],
                "weaken_reward": sample["weaken_reward"]
            })
    
    return results

def run_inference_pipeline(
    model_path: str,
    rollout_path: str,
    output_path: str,
    max_samples: int = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "cuda"
):
    """运行完整的推理流水线"""
    print(f"🔍 Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        "./model/qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )
    
    print(f"🔍 Loading trained adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print(f"🔍 Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "./model/qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token_id = 151643
    processor.tokenizer.eos_token_id = 151645
    
    print(f"🔍 Loading inference dataset...")
    inference_data = load_inference_dataset(rollout_path, max_samples)
    
    if not inference_
        raise ValueError("❌ No valid samples found for inference!")
    
    print(f"🚀 Starting inference with {len(inference_data)} samples...")
    results = run_inference(
        model=model,
        processor=processor,
        inference_data=inference_data,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    print(f"💾 Saving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    successful = sum(1 for r in results if not r["generated_response"].startswith("ERROR"))
    print(f"\n✅ Inference completed!")
    print(f"   Total samples: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(results) - successful}")
    print(f"   Results saved to: {output_path}")
    
    return results

# ======================
# 🎯 主函数
# ======================
def main():
    parser = argparse.ArgumentParser(description="Multi-Modal DPO Training + Inference")
    parser.add_argument("--mode", choices=["train", "infer", "train_infer"], 
                       default="train", help="Mode: train, infer, or train_infer")
    
    # 训练参数
    parser.add_argument("--rollout_path", type=str, required=True,
                       help="Path to rollout log JSON file")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for training")
    parser.add_argument("--exp_name", type=str, default="dpo_experiment",
                       help="Experiment name")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    
    # 推理参数
    parser.add_argument("--model_path", type=str,
                       help="Path to trained model (for inference mode)")
    parser.add_argument("--inference_output", type=str,
                       help="Path to save inference results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        model_path = run_training(
            exp_name=args.exp_name,
            rollout_path=args.rollout_path,
            output_dir=args.output_dir,
            epochs=args.epochs
        )
        print(f"✅ Training completed! Model saved to: {model_path}")
        
    elif args.mode == "infer":
        if not args.model_path or not args.inference_output:
            raise ValueError("--model_path and --inference_output are required for inference mode")
        
        run_inference_pipeline(
            model_path=args.model_path,
            rollout_path=args.rollout_path,
            output_path=args.inference_output,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
    elif args.mode == "train_infer":
        # 先训练
        model_path = run_training(
            exp_name=args.exp_name,
            rollout_path=args.rollout_path,
            output_dir=args.output_dir,
            epochs=args.epochs
        )
        
        # 再推理
        if not args.inference_output:
            args.inference_output = os.path.join(args.output_dir, args.exp_name, "inference_results.json")
        
        run_inference_pipeline(
            model_path=model_path,
            rollout_path=args.rollout_path,
            output_path=args.inference_output,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    main()