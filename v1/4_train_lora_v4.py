#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal Diversity DPO Trainer for OpenFly
✅ 多模态输入 | ✅ 多样性正则化 | ✅ LoRA 微调
"""

import os
import json
import torch
import argparse
import numpy as np
from typing import Dict, List, Any
from datasets import Dataset, Features, Sequence, Value
from transformers import TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor

from training.multimodal_collator import MultimodalDPODataCollator
# from training.DiversityDPOTrainer import DiversityDPOTrainer
from training.DiversityDPOTrainer import MultimodalDiversityDPOCollator
from training.DiversityDPOTrainer import EnhancedDiversityDPOTrainer


from trl import DPOTrainer

original_get_batch_samples = DPOTrainer.get_batch_samples

def patched_get_batch_samples(self, dataloader, num_batches, device=None):
    return original_get_batch_samples(self, dataloader, num_batches)

DPOTrainer.get_batch_samples = patched_get_batch_samples

# ======================
# 📊 多模态 DPO 数据集构建
# ======================

def select_dpo_pairs(candidates: List[Dict], min_gap: float = 0.2) -> List[tuple]:
    """
    智能选择 DPO 配对策略
    Returns: [(chosen, rejected, reward_gap), ...]
    """
    candidates.sort(key=lambda x: x["reward"], reverse=True)
    
    pairs = []
    
    # 1. 首选：最高 vs 最差（如果 gap 足够大）
    max_reward = candidates[0]["reward"]
    min_reward = candidates[-1]["reward"]
    gap_max_min = max_reward - min_reward
    
    if gap_max_min >= min_gap:
        pairs.append((candidates[0], candidates[-1], gap_max_min))
        print(f"✅ 使用 max-min 配对: gap={gap_max_min:.3f}")
    else:
        # 2. 备选：最高 vs 所有其他（多对训练）
        for i in range(1, len(candidates)):
            gap = max_reward - candidates[i]["reward"]
            if gap >= min_gap * 0.5:  # 放宽条件
                pairs.append((candidates[0], candidates[i], gap))
                print(f"✅ 使用 max-vs-{i} 配对: gap={gap:.3f}")
    
    # 3. 如果还是没有，用最高 vs 次高（强制训练）
    if not pairs and len(candidates) >= 2:
        gap = candidates[0]["reward"] - candidates[1]["reward"]
        pairs.append((candidates[0], candidates[1], gap))
        print(f"⚠️ 强制使用 max-second 配对: gap={gap:.3f}")
    
    return pairs

def build_multimodal_dpo_dataset(rollout_path: str) -> Dataset:
    """构建最优 DPO 数据集"""
    with open(rollout_path, 'r') as f:
        data = json.load(f)
    
    print(f"📥 Loaded {len(data)} rollout samples")
    
    dpo_data = []
    min_reward_gap = 0.15  # 最小可接受 gap
    
    for i, item in enumerate(data):
        # 提取基本信息
        image_paths = item.get("image_paths", [])
        weaken = item.get("weaken_instruction", "").strip()
        
        if not weaken or len(weaken) < 5:
            continue
        
        # 提取候选翻译
        candidates = []
        for j in range(1, 4):
            text_key = f"translated_instruction_{j}"
            reward_key = f"{text_key}_reward"
            
            if text_key in item and reward_key in item:
                try:
                    text = str(item[text_key]).strip()
                    reward = float(item[reward_key])
                    if text and len(text) > 5 and 0 <= reward <= 1:
                        candidates.append({"text": text, "reward": reward, "id": j})
                except (ValueError, TypeError, KeyError):
                    continue
        
        if len(candidates) < 2:
            continue
        
        # 🔑 关键：智能配对选择
        pairs = select_dpo_pairs(candidates, min_gap=min_reward_gap)
        
        for chosen_candidate, rejected_candidate, reward_gap in pairs:
            dpo_data.append({
                "images": image_paths,
                "prompt": weaken,
                "chosen": chosen_candidate["text"],
                "rejected": rejected_candidate["text"],
                "reward_gap": reward_gap,
                "pair_type": "max-min" if reward_gap >= min_reward_gap else "max-second"
            })
    
    print(f"✅ Optimal DPO dataset: {len(dpo_data)} pairs")
    if dpo_data :
        gaps = [item["reward_gap"] for item in dpo_data]
        pair_types = [item["pair_type"] for item in dpo_data]
        print(f"   Reward gap - min: {min(gaps):.3f}, max: {max(gaps):.3f}, avg: {sum(gaps)/len(gaps):.3f}")
        print(f"   Pair types: max-min={pair_types.count('max-min')}, max-second={pair_types.count('max-second')}")
    
    # 创建 Dataset
    from datasets import Dataset, Features, Sequence, Value
    if dpo_data:
        features = Features({
            "images": Sequence(Value("string")),
            "prompt": Value("string"),
            "chosen": Value("string"),
            "rejected": Value("string"),
            "reward_gap": Value("float32"),
            "pair_type": Value("string")
        })
        return Dataset.from_list(dpo_data, features=features)
    else:
        # 返回最小数据集用于调试
        raise ValueError("No valid DPO pairs generated!")
    
from training.multimodal_collator import MultimodalDPODataCollator
from training.DiversityDPOTrainer import EnhancedDiversityDPOTrainer
# ======================
# 📈 训练监控回调
# ======================
class TrainingMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            if step % 50 == 0:
                loss = logs.get("loss", 0)
                lr = logs.get("learning_rate", 0)
                print(f"📈 Step {step}: loss={loss:.4f}, lr={lr:.2e}")


def configure_lora(model) -> LoraConfig:
    """安全配置 LoRA"""
    
    # 方案1: 动态检测（最可靠）
    target_modules = []
    
    # 检查模型结构
    try:
        num_layers = len(model.language_model.layers)
        print(f"🔍 Detected {num_layers} transformer layers")
        
        for i in range(num_layers):
            base_name = f"language_model.layers.{i}.self_attn"
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                full_name = f"{base_name}.{proj}"
                try:
                    _ = model.get_submodule(full_name)
                    target_modules.append(full_name)
                except AttributeError:
                    continue
        
        if len(target_modules) == 0:
            raise ValueError("No target modules found")
            
        print(f"✅ Configured {len(target_modules)} LoRA modules")
        return LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=target_modules,
            lora_dropout=0.15,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
    except Exception as e:
        print(f"❌ Auto-detection failed: {e}")
        print("🔄 Falling back to regex pattern...")
        
        # 方案2: 使用转义的正则表达式
        return LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[
                r"language_model\.layers\..*\.self_attn\.q_proj",
                r"language_model\.layers\..*\.self_attn\.k_proj",
                r"language_model\.layers\..*\.self_attn\.v_proj", 
                r"language_model\.layers\..*\.self_attn\.o_proj",
            ],
            lora_dropout=0.15,
            bias="none",
            task_type="CAUSAL_LM"
        )

# ======================
# 🚀 主训练流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_path", type=str, default="./train_t2rl_lora/data/rollout_with_trajectory.json", help="Rollout日志路径")  
    parser.add_argument("--output_dir", type=str, default="./train_t2rl_lora/output/checkpoint")
    parser.add_argument("--diversity_weight", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    args = parser.parse_args()
    
    print("🚀 Starting Multi-Modal Diversity DPO Training")
    
    # 1. 构建数据集
    dpo_dataset = build_multimodal_dpo_dataset(args.rollout_path)
    if len(dpo_dataset) == 0:
        raise ValueError("No valid DPO pairs generated!")
    
    # 2. 加载模型和 processor
    print("🔍 Loading model and processor...")
    model = AutoModelForVision2Seq.from_pretrained(
        "./model/qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(
        "./model/qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    
    # 3. 配置 LoRA（仅语言模型）
    print("🔧 Configuring LoRA...")
    # lora_config = LoraConfig(
    #     r=4,
    #     lora_alpha=8,
    #     target_modules=[
    #         "language_model\.layers\..*\.self_attn\.q_proj",
    #         "language_model\.layers\..*\.self_attn\.k_proj",
    #         "language_model\.layers\..*\.self_attn\.v_proj",
    #         "language_model\.layers\..*\.self_attn\.o_proj",
    #         ],
    #     lora_dropout=0.15,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    
    lora_config = configure_lora(model)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 数据整理器
    collator = MultimodalDiversityDPOCollator(processor, max_images=60)
    
    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
    )
    
    # 6. Trainer
    trainer = EnhancedDiversityDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dpo_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        beta=0.1,  # DPO beta 参数
    )
    
    trainer.add_callback(TrainingMonitorCallback())
    
    # 7. 开始训练
    print(f"🎯 Starting training with {len(dpo_dataset)} samples...")
    trainer.train()
    
    # 8. 保存
    print("💾 Saving model...")
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))
    
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    main()