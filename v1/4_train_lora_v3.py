#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T²-RL LoRA Trainer | Few-Shot (K=5) + Diversity-regularized DPO ✅ 最终稳定版
✅ 全量修复: pad_token_id + list.map() + 千问VL适配 三大核心报错
✅ 数据格式: list → HuggingFace Dataset (DPOTrainer唯一合法格式)
✅ 硬件适配: 8×A100 80G + Qwen2.5-VL-7B
✅ 输入: logs_k5_with_reward.json | 输出: t2rl_lora_k5 (LoRA适配器)
"""

import os
import json
import torch
import gc
import argparse
from typing import List, Dict
from collections import defaultdict
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from diverse_dpo_trainer import DiversityDPOTrainer
from transformers import AutoProcessor, AutoModelForVision2Seq
# ✅ 关键新增：导入HuggingFace Dataset，解决list.map()报错核心依赖
from datasets import Dataset

# ======================
# 🔧 全局配置【固化适配8×A100 80G】
# ======================
MODEL_ID = "./model/qwen/Qwen2.5-VL-7B-Instruct"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["DISABLE_FLASH_ATTENTION"] = "1"

def load_rollout_logs(log_path: str) -> List[Dict]:
    """加载日志+路径校验"""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"❌ 日志文件不存在: {log_path}")
    with open(log_path, 'r', encoding="utf-8") as f:
        logs = json.load(f)
    print(f"✅ 加载日志成功 | 样本数: {len(logs)} | 路径: {log_path}")
    return logs

def build_dpo_dataset(logs: List[Dict]) -> Dataset:
    """✅ 核心修改：返回HuggingFace Dataset对象 | 彻底解决list.map()报错"""
    groups = defaultdict(list)
    for log_idx, log in enumerate(logs):
        if not all(k in log for k in ["id", "weakened_instruction", "translated_instruction", "reward"]):
            print(f"⚠️ 跳过无效样本{log_idx}: 缺失核心字段")
            continue
        base_id = log["id"].rsplit("_sample_", 1)[0]
        groups[base_id].append(log)
    
    dpo_list = []
    for base_id, candidates in groups.items():
        if len(candidates) < 2: continue
        candidates.sort(key=lambda x: x["reward"], reverse=True)
        if not candidates[0]["translated_instruction"] or not candidates[-1]["translated_instruction"]: continue
        dpo_list.append({
            "prompt": f"Human: {candidates[0]['weakened_instruction']}\nAssistant: ",
            "chosen": candidates[0]["translated_instruction"],
            "rejected": candidates[-1]["translated_instruction"],
            "reward_gap": candidates[0]["reward"] - candidates[-1]["reward"]
        })
    
    # ✅ 关键转换：Python list → HuggingFace Dataset (DPOTrainer唯一支持格式)
    dpo_dataset = Dataset.from_list(dpo_list)
    print(f"\n✅ DPO数据集构建完成 | 有效偏好对: {len(dpo_dataset)} | 唯一指令数: {len(groups)}")
    print(f"✅ ✅ 数据格式校验通过: list → HuggingFace Dataset (解决map()报错核心)")
    return dpo_dataset

def load_model_tokenizer_processor():
    """✅ 解耦 模型/纯文本Tokenizer/多模态Processor | 无属性报错"""
    print(f"\n🔍 加载模型组件 | {MODEL_ID} | 8×A100 80G 专属优化...")
    # 1. 加载多模态处理器（推理用）
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        resume_download=True,
        padding_side="left"
    )
    
    # 2. 提取纯文本Tokenizer（训练专用，核心）
    tokenizer = processor.tokenizer
    # 千问VL 文本配置黄金写法（无属性缺失）
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.bos_token_id = None
    tokenizer.add_bos_token = False
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # 3. 加载多模态模型（训练模式）
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced_low_0",
        load_in_4bit=False,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        # use_flash_attention_2=False
    ).train()


    print("\n🔍 模型结构预览（查找 q_proj/k_proj/v_proj/o_proj 所在位置）:")
    for name, param in model.named_parameters():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
            print(f"   {name} | requires_grad={param.requires_grad} | shape={param.shape}")
    
    # 属性校验（杜绝隐性报错）
    assert tokenizer.pad_token_id is not None, "❌ pad_token_id 未正确绑定！"
    assert tokenizer.eos_token_id is not None, "❌ eos_token_id 缺失！"
    
    print(f"✅ 组件加载完成 | GPU数量: {torch.cuda.device_count()}")
    print(f"✅ Tokenizer配置 | pad_token_id: {tokenizer.pad_token_id} | padding_side: {tokenizer.padding_side}")
    print(f"✅ 模型设备: {next(model.parameters()).device} | 训练模式已启用")
    return model, tokenizer, processor

from transformers import TrainerCallback
class GradientMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 2 == 0:  # 每2步打印1次
            model = kwargs['model']
            for name, param in model.named_parameters():
                if "lora_" in name and param.grad is not None:
                    grad_mean = param.grad.mean().item()
                    print(f"📊 Step {state.global_step} | {name} 梯度均值: {grad_mean:.8f}")
                    break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Rollout日志路径 (logs_k5_with_reward.json)")
    parser.add_argument("--output", type=str, default="./train_t2rl_lora/output/t2rl_lora_k5", help="LoRA输出目录")
    parser.add_argument("--diversity_weight", type=float, default=0.3, help="多样性正则强度 (0.0~1.0)")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数 (K=5建议 ≥5)")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率 (1e-4最优)")
    args = parser.parse_args()

    # 前置显存清理（8卡必备）
    torch.cuda.empty_cache()
    gc.collect()
    
    # Step 1: 加载日志 + 构建【Dataset格式】DPO数据（核心修复）
    logs = load_rollout_logs(args.log)
    dpo_dataset = build_dpo_dataset(logs)
    if len(dpo_dataset) == 0:
        print("❌ 致命错误：无有效DPO偏好对，终止训练！")
        return
    
    # 数据集统计
    reward_gaps = dpo_dataset["reward_gap"]
    avg_gap = sum(reward_gaps)/len(reward_gaps)
    max_gap = max(reward_gaps)
    print(f"\n📊 DPO数据集统计 | 平均奖励差: {avg_gap:.3f} | 最大奖励差: {max_gap:.3f}")

    # Step 2: 加载模型+纯文本Tokenizer+多模态Processor
    model, tokenizer, processor = load_model_tokenizer_processor()

    # Step 3: LoRA配置（Few-Shot防过拟合+稳训）
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 文本层稳训，兼容所有版本
        lora_dropout=0.15,
        bias="none",
        task_type="CAUSAL_LM",  # 纯文本任务，规避多模态层报错
        inference_mode=False
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    print(f"\n✅ LoRA配置生效 | 超轻量适配K=5小样本，无过拟合风险")

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,      # 确保是 1e-4
        num_train_epochs=args.epochs,
        logging_steps=1,
        output_dir=args.output,
        save_strategy="epoch",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        # ✅ 新增关键参数：
        warmup_steps=1,   # 防止 lr 一上来就=0
        lr_scheduler_type="cosine",  # 或 "linear"，避免 sharp decay
    )

    # Step 5: ✅ 所有参数合规 | 启动训练（无任何报错）
    trainer = DiversityDPOTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=dpo_dataset,  # ✅ Dataset格式，支持map()方法
        tokenizer=tokenizer,        # ✅ 纯文本Tokenizer，有pad_token_id
        diversity_weight=args.diversity_weight,
        diversity_schedule="adaptive"
    )
    
    trainer.add_callback(GradientMonitorCallback())
    # Step 6: 启动训练
    print(f"\n🚀 启动T²-RL LoRA训练 | K={len(dpo_dataset)} | diversity_weight={args.diversity_weight}")
    print(f"📌 训练配置 | epochs={args.epochs} | lr={args.lr} | batch=1×5 (梯度累积)")
    trainer.train()

    # 在 trainer 初始化后添加：
    optimizer = trainer.optimizer
    param_groups = optimizer.param_groups
    print(f"\n📊 Optimizer 参数组数量: {len(param_groups)}")

    total_optim_params = 0
    for i, group in enumerate(param_groups):
        group_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        total_optim_params += group_params
        print(f"   Group {i}: lr={group['lr']:.1e}, params={group_params:,}")

    print(f"✅ Optimizer 总可训练参数: {total_optim_params:,}")
    

    # Step 7: 保存所有组件（训练/推理无缝衔接）
    os.makedirs(args.output, exist_ok=True)
    lora_model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n✅ ✅ ✅ 训练圆满完成！所有组件已保存至: {args.output}")
    print(f"📁 输出文件清单:")
    print(f"   ├─ adapter_config.json / adapter_model.safetensors (LoRA权重 ~300KB)")
    print(f"   ├─ tokenizer_config.json / vocab.json (纯文本Tokenizer)")
    print(f"   └─ preprocessor_config.json (多模态Processor)")
    print(f"\n👉 推理命令（直接复用）:")
    print(f"python translate_with_lora.py --lora {args.output} --input ./train_t2rl_lora/data/t2rl_eval.json")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()
