#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T²-RL LoRA Trainer | Few-Shot (K=5) + Diversity-regularized DPO
✅ 输入: logs_k5_with_reward.json
✅ 输出: output/t2rl_lora/ (LoRA 适配器)
"""

import os
import json
import torch
import argparse
from typing import List, Dict
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from diverse_dpo_trainer import DiversityDPOTrainer  # 确保已放置同目录

from transformers import AutoProcessor, AutoModelForVision2Seq



def load_rollout_logs(log_path: str) -> List[Dict]:
    """加载 rollout 日志"""
    with open(log_path, 'r') as f:
        logs = json.load(f)
    print(f"✅ Loaded {len(logs)} rollout samples from {log_path}")
    return logs

def build_dpo_dataset(logs: List[Dict]) -> List[Dict]:
    """构建 DPO 偏好对"""
    # 按 weakened_instruction 分组（忽略 sample_id）
    groups = defaultdict(list)
    for log in logs:
        # 提取 base_id: "traj_001_sample_1" → "traj_001"
        base_id = log["id"].rsplit("_sample_", 1)[0]
        groups[base_id].append(log)
    
    dpo_data = []
    for base_id, candidates in groups.items():
        # 按 reward 降序排序
        candidates.sort(key=lambda x: x["reward"], reverse=True)
        
        # 构建偏好对：最高 reward vs 最低 reward
        if len(candidates) >= 2:
            dpo_data.append({
                "prompt": f"Human: {candidates[0]['weakened_instruction']}\nAssistant: ",
                "chosen": candidates[0]["translated_instruction"],
                "rejected": candidates[-1]["translated_instruction"],
                "reward_gap": candidates[0]["reward"] - candidates[-1]["reward"]
            })
    
    print(f"✅ Built {len(dpo_data)} DPO pairs from {len(groups)} unique instructions")
    return dpo_data

# # ======================
# # 🤖 模型加载【Qwen2.5-VL-7B 完美适配版】
# # ======================
# def load_model_and_processor():
#     print(f"\n🔍 加载模型 {MODEL_ID} | 适配8×A100 80G集群（7B专属优化）...")
#     processor = AutoProcessor.from_pretrained(
#         MODEL_ID,
#         trust_remote_code=True,
#         resume_download=True
#     )

#     model = AutoModelForVision2Seq.from_pretrained(
#         MODEL_ID,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#         device_map="balanced_low_0",
#         load_in_4bit=False,
#         low_cpu_mem_usage=True,
#         ignore_mismatched_sizes=True
#     ).eval()
    
#     print(f"✅ 模型加载完成 | GPU数量: {torch.cuda.device_count()} | 推理模式已启用")
#     return processor, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True,
                        help="Rollout日志路径 (e.g., logs_k5_with_reward.json)")
    parser.add_argument("--output", type=str, default="output/t2rl_lora",
                        help="LoRA输出目录")
    parser.add_argument("--diversity_weight", type=float, default=0.3,
                        help="多样性正则强度 (0.0~1.0)")
    args = parser.parse_args()

    # Step 1: 加载日志 + 构建 DPO 数据
    logs = load_rollout_logs(args.log)
    dpo_data = build_dpo_dataset(logs)
    
    # 检查数据
    if len(dpo_data) == 0:
        print("❌ Error: No DPO pairs built! Check log format.")
        return
    print(f"📊 DPO Dataset Stats:")
    print(f"   - Avg Reward Gap: {sum(p['reward_gap'] for p in dpo_data)/len(dpo_data):.3f}")
    print(f"   - Max Reward Gap: {max(p['reward_gap'] for p in dpo_data):.3f}")

    # Step 2: 加载 Qwen-VL 模型
    print("\n🔍 Loading Qwen-VL-Chat...")
    model_name_or_path = './model/qwen/Qwen2.5-VL-7B-Instruct'  # 本地模型路径
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name_or_path,
    #     trust_remote_code=True,
    #     padding_side="left"
    # )
    # tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    #     use_flash_attention_2=True
    # )
    # ========== 关键修复1：增加 trust_remote_code=True + 适配千问的特殊配置 ==========
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,  # 必须开启，加载模型自定义的Tokenizer类
        padding_side="left",
        add_eos_token=True,      # 千问模型推荐配置
        add_bos_token=False      # 千问模型专属：禁止添加bos_token，否则会报错
    )
    # 千问模型pad_token必须手动指定为eos_token，固定写法
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 二次确认，避免生成时attention mask错误

    # ========== 关键修复2：模型加载适配千问VL的显存/精度配置 ==========
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,  # 必须开启，加载千问自定义的模型架构
        device_map="auto",       # 自动分配显存到GPU/CPU
        torch_dtype=torch.bfloat16,
        # use_flash_attention_2=True,
        # low_cpu_mem_usage=True,  # 千问大模型必加，降低CPU内存占用
    ).train()
    


    # 加载完成验证
    print("✅ Qwen2.5-VL-7B模型与Tokenizer加载成功！")
    print(f"Tokenizer类别: {type(tokenizer)}")
    print(f"模型设备: {model.device}")

    # Step 3: 配置 LoRA (超轻量)
    lora_config = LoraConfig(
        r=4,                    # ↓ Few-Shot 用 r=4 (300K 参数)
        lora_alpha=8,
        target_modules=["wqkv", "wo"],  # Qwen-VL attention 层
        lora_dropout=0.1,       # ↑ 防止 Few-Shot 过拟合
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            param.data.required_grad = True
    
    model.train()
    model.enable_input_require_grads()  # 启用输入梯度（DPO需要）
    model.print_trainable_parameters()
    print(f"✅ LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}")

    # Step 4: 配置 Diversity DPO Trainer
    trainer = DiversityDPOTrainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # K=5 用 batch=1
            gradient_accumulation_steps=5,  # 等效 batch=5
            learning_rate=1e-4,
            num_train_epochs=1,
            max_grad_norm=0.5,
            logging_steps=1,
            output_dir=args.output,
            save_strategy="epoch",
            report_to="none"
        ),
        train_dataset=dpo_data,
        tokenizer=tokenizer,
        # ✅ Diversity Regularization
        diversity_weight=args.diversity_weight,
        diversity_schedule="adaptive"
    )

    
    
    # Step 5: 训练
    print(f"\n🚀 Starting LoRA Training (K={len(dpo_data)}, diversity_weight={args.diversity_weight})...")
    trainer.train()
    
    # Step 6: 保存 LoRA
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n✅ LoRA saved to: {args.output}")
    print(f"   - Files: adapter_config.json, adapter_model.safetensors (~300KB)")
    print(f"   - Next: python translate_with_lora.py --lora {args.output} --input data/t2rl_eval.json")

if __name__ == "__main__":
    main()


# 训练 Diversity LoRA (K=5)
# python 4_train_lora.py --log ./train_t2rl_lora/data/logs_k5_with_reward.json --output ./train_t2rl_lora/output/t2rl_lora_k5 --diversity_weight 0.3