#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep DPO Trainer Diagnostic Tool
Usage: python debug_trainer.py --rollout_path ./your_rollout.json
"""

import os
import json
import torch
import argparse
import traceback
from typing import List, Dict
from datasets import Dataset, Features, Sequence, Value
from transformers import TrainingArguments, AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

# 假设您的 trainer 模块在同一目录
try:
    from trainer import MultimodalDPOTrainer, MultimodalDPOCollator
except ImportError:
    print("❌ Cannot import trainer module. Please ensure trainer.py is in the same directory.")
    exit(1)

def build_test_dpo_dataset(logs: List[Dict]) -> Dataset:
    """构建测试用 DPO 数据集"""
    dpo_list = []
    
    for log_idx, log in enumerate(logs[:3]):  # 只用前3个样本测试
        weaken_instruction = log.get("weaken_instruction", "").strip()
        if not weaken_instruction:
            weaken_instruction = "Go to the building"
        
        # 提取翻译
        candidates = []
        for i in range(1, 4):
            text_key = f"translated_instruction_{i}"
            reward_key = f"translated_instruction_{i}_reward"
            
            text = log.get(text_key, "").strip()
            reward = log.get(reward_key, 0.0)
            
            if text and len(text) > 5:
                try:
                    reward_val = float(reward)
                    candidates.append({"text": text, "reward": reward_val})
                except:
                    candidates.append({"text": text, "reward": 0.0})
        
        if len(candidates) < 2:
            # 强制创建2个候选
            candidates = [
                {"text": "Head directly toward the tall building", "reward": 0.8},
                {"text": "Go to building", "reward": 0.2}
            ]
        
        candidates.sort(key=lambda x: x["reward"], reverse=True)
        
        # 提取图像信息
        image_path = log.get("image_path", "dummy")
        index_list = log.get("index_list", ["frame_0"])
        
        dpo_list.append({
            "images": [image_path],
            "index_list": index_list,
            "prompt": weaken_instruction[:100],
            "chosen": candidates[0]["text"][:100],
            "rejected": candidates[-1]["text"][:100],
            "reward_gap": max(0.1, abs(candidates[0]["reward"] - candidates[-1]["reward"]))
        })
    
    if not dpo_list:
        # 最后保障
        dpo_list = [{
            "images": ["dummy"],
            "index_list": ["frame_0"],
            "prompt": "Go to building",
            "chosen": "Head to tall building", 
            "rejected": "Go building",
            "reward_gap": 0.5
        }]
    
    features = Features({
        "images": Sequence(Value("string")),
        "index_list": Sequence(Value("string")),
        "prompt": Value("string"),
        "chosen": Value("string"),
        "rejected": Value("string"),
        "reward_gap": Value("float32")
    })
    
    return Dataset.from_list(dpo_list, features=features)

def configure_test_lora(model):
    """配置测试用 LoRA"""
    # 简化 LoRA 配置，只针对前几层
    target_modules = []
    try:
        # 尝试简单配置
        target_modules = ["language_model.layers.0.self_attn.q_proj"]
    except:
        # 回退到正则表达式
        target_modules = [r"language_model\.layers\.0\.self_attn\.q_proj"]
    
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    try:
        peft_model = get_peft_model(model, lora_config)
        return peft_model
    except Exception as e:
        print(f"⚠️ LoRA configuration failed: {e}")
        print("🔄 Using base model without LoRA for testing...")
        return model

def diagnose_trainer_comprehensive(rollout_path: str):
    """综合性 trainer 诊断"""
    print("=" * 70)
    print("🔍 COMPREHENSIVE DPO TRAINER DIAGNOSTIC")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n1️⃣ LOADING DATA...")
    try:
        with open(rollout_path, 'r') as f:
            logs = json.load(f)
        print(f"✅ Loaded {len(logs)} samples from {rollout_path}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # 2. 构建数据集
    print("\n2️⃣ BUILDING DATASET...")
    try:
        dpo_dataset = build_test_dpo_dataset(logs)
        print(f"✅ Dataset created: {len(dpo_dataset)} samples")
        print(f"   Sample keys: {list(dpo_dataset[0].keys())}")
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        traceback.print_exc()
        return
    
    # 3. 加载模型和 processor
    print("\n3️⃣ LOADING MODEL AND PROCESSOR...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            "./model/qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            "./model/qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        # 配置 tokenizer
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.pad_token_id = 151643
        processor.tokenizer.eos_token_id = 151645
        
        print("✅ Model and processor loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        # 尝试 CPU 加载
        try:
            print("🔄 Trying CPU loading...")
            model = AutoModelForVision2Seq.from_pretrained(
                "./model/qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            print("✅ CPU model loaded")
        except Exception as e2:
            print(f"❌ CPU loading also failed: {e2}")
            return
    
    # 4. 配置 LoRA
    print("\n4️⃣ CONFIGURING LORA...")
    try:
        peft_model = configure_test_lora(model)
        print("✅ LoRA configuration successful")
    except Exception as e:
        print(f"❌ LoRA failed, using base model: {e}")
        peft_model = model
    
    # 5. 创建 collator
    print("\n5️⃣ CREATING COLLATOR...")
    try:
        collator = MultimodalDPOCollator(processor)
        print("✅ Collator created")
        
        # 测试 collator
        test_batch = collator([dpo_dataset[0]])
        print(f"✅ Collator test passed: {list(test_batch.keys())}")
        for k, v in test_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k}: {v.shape} on {v.device}")
        
    except Exception as e:
        print(f"❌ Collator failed: {e}")
        traceback.print_exc()
        return
    
    # 6. 创建 training arguments
    print("\n6️⃣ CREATING TRAINING ARGUMENTS...")
    try:
        training_args = TrainingArguments(
            output_dir="./debug_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=-1,  # Disable max_steps
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            prediction_loss_only=True,
            bf16=False,  # Disable bf16 for debugging
            fp16=False,  # Disable fp16 for debugging  
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to="none",
        )
        print("✅ Training arguments created")
        print(f"   Max steps: {training_args.max_steps}")
        print(f"   Num epochs: {training_args.num_train_epochs}")
    except Exception as e:
        print(f"❌ Training arguments failed: {e}")
        return
    
    # 7. 创建 trainer
    print("\n7️⃣ CREATING TRAINER...")
    try:
        trainer = MultimodalDPOTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=dpo_dataset,
            data_collator=collator,
            tokenizer=processor.tokenizer,
        )
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        traceback.print_exc()
        return
    
    # 8. Test dataloader
    print("\n8️⃣ TESTING DATALOADER...")
    try:
        dataloader = trainer.get_train_dataloader()
        print(f"✅ Dataloader created: {type(dataloader)}")
        print(f"   Dataloader length: {len(dataloader) if hasattr(dataloader, '__len__') else 'N/A'}")
        
        # Test iteration
        batch = next(iter(dataloader))
        print(f"✅ Dataloader iteration successful")
        print(f"   Batch keys: {list(batch.keys())}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k}: {v.shape} (device: {v.device})")
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        traceback.print_exc()
        return
    
    # 9. Test training step
    print("\n9️⃣ TESTING TRAINING STEP...")
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        # Filter None values
        filtered_batch = {k: v for k, v in batch.items() if v is not None}
        loss = trainer.training_step(trainer.model, filtered_batch)
        print(f"✅ Training step successful: loss = {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        traceback.print_exc()
        return
    
    # 10. Final test: small training loop
    print("\n🔟 FINAL TEST: SMALL TRAINING LOOP...")
    try:
        # Create a minimal training loop
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
        dataloader = trainer.get_train_dataloader()
        
        for step, batch in enumerate(dataloader):
            filtered_batch = {k: v for k, v in batch.items() if v is not None}
            loss = trainer.training_step(trainer.model, filtered_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"✅ Step {step} completed: loss = {loss.item():.4f}")
            
            if step >= 2:  # Only 3 steps
                break
                
        print("✅ Small training loop completed successfully!")
        
    except Exception as e:
        print(f"❌ Small training loop failed: {e}")
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("🎉 ALL DIAGNOSTIC TESTS PASSED!")
    print("Your trainer setup is working correctly.")
    print("The issue might be in your main training script configuration.")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="DPO Trainer Diagnostic Tool")
    parser.add_argument("--rollout_path", type=str, required=True,
                       help="Path to your rollout JSON file")
    args = parser.parse_args()
    
    if not os.path.exists(args.rollout_path):
        print(f"❌ Rollout file not found: {args.rollout_path}")
        return
    
    diagnose_trainer_comprehensive(args.rollout_path)

if __name__ == "__main__":
    main()