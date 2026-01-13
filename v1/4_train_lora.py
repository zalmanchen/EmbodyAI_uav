#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal DPO Training Script
✅ Baseline vs Diversity Comparison
"""

import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
import torch
import argparse
import warnings
from PIL import Image
from io import BytesIO
from datasets import Dataset, Features, Sequence, Value
from transformers import TrainingArguments, AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import pandas as pd

# 导入 trainer 模块
from training.trainer import *

# ======================
# 📊 数据集构建
# ======================

from typing import List, Dict
from collections import defaultdict


def increment_version_and_mkdir(path_str: str, version_prefix: str = "_v") -> str:
    """
    自动加版本号 + 检测路径是否存在 + 创建目录
    Args:
        path_str: 原始路径
        version_prefix: 版本号前缀
    Returns:
        可用的新路径（确保目录不存在，自动创建）
    """
    def get_new_path(original_path):
        """内部函数：单次版本号加1"""
        pattern = re.compile(f"{version_prefix}(\\d+)")
        match = pattern.search(original_path)
        if not match:
            return original_path + version_prefix + "1"
        else:
            version_num = int(match.group(1))
            version_str = match.group(1)
            new_version_num = version_num + 1
            new_version_str = f"{new_version_num:0{len(version_str)}d}"
            return pattern.sub(f"{version_prefix}{new_version_str}", original_path)
    
    # 循环检测：若新路径已存在，继续加1
    new_path = get_new_path(path_str)
    while os.path.exists(new_path):
        new_path = get_new_path(new_path)
    
    # 创建目录（支持多层级）
    os.makedirs(new_path, exist_ok=True)
    
    return new_path




DATASET_ROOT_PATH = "../data/openfly/traj"
TEMP_IMAGE_DIR = './tmp/qwen_vl_imgs'
TARGET_SIZE = (448, 448)

version = 2
model_path = f"./model/qwen/Qwen2.5-VL-7B-Instruct_v2"
rollout_path = f'./train_t2rl_lora/data_v2'

save_model_path = increment_version_and_mkdir(model_path, "-v")


def build_multimodal_dpo_dataset(rollout_path: str) -> Dataset:
    """构建多模态 DPO 数据集"""
    with open(rollout_path, 'r') as f:
        logs = json.load(f)

    """
    ✅ 适配您的数据格式的多模态 DPO 数据集构建
    ✅ 每个样本生成多对 (chosen, rejected) 以增加数据量
    """
    dpo_list = []
    
    for log_idx, log in enumerate(logs):
        print(f"🔍 Processing sample {log_idx}: {log.get('image_path', 'N/A')}")
        
        # 提取基本信息
        image_path = log.get("image_path", "")
        index_list = log.get("index_list", [])
        weaken_instruction = log.get("weaken_instruction", "").strip()
        weaken_reward = log.get("weaken_instruction_reward") # get weaken_reward 
        
        if not weaken_instruction or not index_list:
            print(f"  ⚠️ Skip sample {log_idx}: missing instruction or frames")
            continue
        
        # 提取 3 种翻译及其奖励
        candidates = []
        for i in range(1, 5):
            text_key = f"translated_instruction_{i}"
            reward_key = f"translated_instruction_{i}_reward"
            
            if text_key in log and reward_key in log:
                text = log[text_key].strip()
                try:
                    reward = float(log[reward_key])
                    if text and len(text) > 10:  # 过滤短文本
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
        
        # 创建tmp image and tmp iamge pahth
        image_paths = []

        _temp_path = os.path.join(TEMP_IMAGE_DIR, image_path)
        os.makedirs(_temp_path, exist_ok=True)

        # import pdb; pdb.set_trace()

        # read parquet stored in 
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

        

        # 🔑 关键：构建多对 DPO 样本（增强数据）
        # 策略1: 最高 vs 最低（强对比）
        if len(candidates) >= 2:
            dpo_list.append({
                "images": image_paths,  # 暂时只存路径，collator 会处理
                "prompt": weaken_instruction,
                "chosen": candidates[0]["text"],
                "rejected": candidates[-1]["text"],
                "reward_gap": candidates[0]["reward"] - candidates[-1]["reward"],
                "pair_type": "max-min"
            })
        
        # 策略2: 最高 vs 次高（细粒度）
        if len(candidates) >= 3:
            dpo_list.append({
                "images": image_paths,
                "prompt": weaken_instruction,
                "chosen": candidates[0]["text"],
                "rejected": candidates[1]["text"],
                "reward_gap": candidates[0]["reward"] - candidates[1]["reward"],
                "pair_type": "max-second"
            })
        
        # 策略3: 次高 vs 最低（中等对比）
        if len(candidates) >= 3:
            dpo_list.append({
                "images": image_paths,
                "prompt": weaken_instruction,
                "chosen": candidates[1]["text"],
                "rejected": candidates[-1]["text"],
                "reward_gap": candidates[1]["reward"] - candidates[-1]["reward"],
                "pair_type": "second-min"
            })
    
    # ✅ 创建 Dataset
    if not dpo_list:
        raise ValueError("❌ No valid DPO pairs generated!")
    
    dpo_dataset = Dataset.from_list(dpo_list)
    print(f"\n✅ Multimodal DPO dataset built: {len(dpo_dataset)} pairs")
    print(f"   Sample types: max-min={sum(1 for x in dpo_list if x['pair_type']=='max-min')}, "
          f"max-second={sum(1 for x in dpo_list if x['pair_type']=='max-second')}")
    
    # 预览
    if len(dpo_dataset) > 0:
        sample = dpo_dataset[0]
        print(f"   First sample preview:")
        print(f"     prompt: {sample['prompt'][:50]}...")
        print(f"     chosen: {sample['chosen'][:50]}...")
        print(f"     rejected: {sample['rejected'][:50]}...")
        print(f"     images: 1 path, {len(sample['images'])} frames")

    return dpo_dataset


# ======================
# 🔧 LoRA 配置
# ======================
def configure_lora(model, target_modules=None):
    """配置 LoRA"""
    if target_modules is None:
        # 动态检测目标模块
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
            # 回退到正则表达式
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度诊断 DPOTrainer 问题
"""

import torch
from transformers import TrainingArguments


def diagnose_trainer_issue(dpo_dataset, collator, processor, peft_model):
    """深度诊断 trainer 问题"""
    
    print("=" * 60)
    print("🔍 DEEP TRAINER DIAGNOSIS")
    print("=" * 60)
    
    # 1. 基础信息
    print(f"1. Dataset info:")
    print(f"   - Length: {len(dpo_dataset)}")
    print(f"   - Type: {type(dpo_dataset)}")
    
    # 2. 创建最简 TrainingArguments
    training_args = TrainingArguments(
        output_dir="./debug_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_steps=-1,  # 禁用 max_steps
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
    )
    
    print(f"\n2. TrainingArgs info:")
    print(f"   - num_train_epochs: {training_args.num_train_epochs}")
    print(f"   - max_steps: {training_args.max_steps}")
    print(f"   - per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"   - gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    
    # 3. 手动创建 DataLoader 并测试
    print(f"\n3. Manual DataLoader test:")
    from torch.utils.data import DataLoader
    manual_dataloader = DataLoader(
        dpo_dataset,
        batch_size=1,
        collate_fn=collator,
        shuffle=False
    )
    
    print(f"   - Manual dataloader length: {len(manual_dataloader)}")
    
    # 测试第一个 batch
    try:
        first_batch = next(iter(manual_dataloader))
        print(f"   - First batch keys: {list(first_batch.keys())}")
        for k, v in first_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"     {k}: {v.shape} on {v.device}")
            else:
                print(f"     {k}: {type(v)} = {v}")
    except Exception as e:
        print(f"   ❌ Manual dataloader failed: {e}")
        return
    
    # 4. 创建 trainer 并深度检查
    print(f"\n4. Creating DPOTrainer...")
    trainer = MultimodalDPOTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dpo_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    
    print(f"   - Trainer created successfully")
    print(f"   - Trainer model type: {type(trainer.model)}")
    print(f"   - Trainer args max_steps: {trainer.args.max_steps}")
    print(f"   - Trainer args num_train_epochs: {trainer.args.num_train_epochs}")
    
    # 5. 检查 trainer 内部的 dataloader
    print(f"\n5. Trainer internal dataloader test:")
    try:
        trainer_dataloader = trainer.get_train_dataloader()
        print(f"   - Trainer dataloader type: {type(trainer_dataloader)}")
        print(f"   - Trainer dataloader has __len__: {hasattr(trainer_dataloader, '__len__')}")
        
        if hasattr(trainer_dataloader, '__len__'):
            print(f"   - Trainer dataloader length: {len(trainer_dataloader)}")
        
        # 尝试获取第一个 batch
        trainer_batch = next(iter(trainer_dataloader))
        print(f"   - Trainer first batch keys: {list(trainer_batch.keys())}")
        
    except Exception as e:
        print(f"   ❌ Trainer dataloader failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 检查 trainer 的 epoch_iterator
    print(f"\n6. Checking trainer epoch_iterator:")
    try:
        # 模拟 trainer 的内部逻辑
        from transformers.trainer import Trainer
        
        # 手动调用 _get_train_sampler
        train_sampler = trainer._get_train_sampler()
        print(f"   - Train sampler: {train_sampler}")
        
        # 手动创建 epoch_iterator
        epoch_iterator = trainer.get_train_dataloader()
        print(f"   - Epoch iterator created: {epoch_iterator}")
        print(f"   - Epoch iterator length: {len(epoch_iterator) if hasattr(epoch_iterator, '__len__') else 'N/A'}")
        
        # 尝试迭代
        batch_count = 0
        for batch in epoch_iterator:
            batch_count += 1
            print(f"   - Successfully iterated batch {batch_count}")
            if batch_count >= 2:
                break
        print(f"   - Total batches iterated: {batch_count}")
        
    except Exception as e:
        print(f"   ❌ Epoch iterator failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 最终测试：手动调用 training_step
    print(f"\n7. Testing training_step directly:")
    try:
        first_batch = next(iter(trainer.get_train_dataloader()))
        loss = trainer.training_step(trainer.model, first_batch)
        print(f"   - training_step successful, loss: {loss}")
        print(f"   - Loss type: {type(loss)}, value: {loss.item()}")
        
    except Exception as e:
        print(f"   ❌ training_step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Trainer should work normally")
    print("=" * 60)
    
    return trainer


# ======================
# 🚀 主训练函数
# ======================
def run_experiment(
    exp_name: str,
    rollout_path: str,
    output_dir: str,
    diversity_weight: float = 0.0,
    diversity_type: str = "cosine",
    epochs: int = 3,
    **trainer_kwargs
    ):
    
    """运行单个实验"""
    print(f"\n🚀 Starting experiment: {exp_name}")
    
    # 1. 构建数据集
    dpo_dataset = build_multimodal_dpo_dataset(rollout_path)

    # if len(dpo_dataset) > 0:
    #     sample = dpo_dataset[0]
    #     print(f"  Sample type: {type(sample)}")
    #     print(f"  Sample keys: {list(sample.keys())}")
    #     for k, v in sample.items():
    #         print(f"    {k}: {type(v)} = {str(v)[:50]}...")
    
    
    # 2. 加载模型
    print("🔍 Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 3. 配置 LoRA
    peft_model = configure_lora(model)
    
    # 4. 处理器
    print("🔍 Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 151643  # Qwen2.5-VL 的 pad token ID
    tokenizer.eos_token_id = 151645  # Qwen2.5-VL 的 eos token ID
    tokenizer.bos_token_id = 151644  # Qwen2.5-VL 的 bos token ID
    print(f"✅ Tokenizer configured: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")
        
    # 5. 训练配置
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, exp_name),
        num_train_epochs=5,    # epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1, # save 3 times per epoch
        save_total_limit=2,
        max_steps=-1, # keys: 禁止使用 max_steps
        
        # 🔑 关键：禁用多模态不支持的功能
        eval_strategy="no",
        # evaluation_strategy="no",  # older versions
        # predict_with_generate=False,
        include_inputs_for_metrics=False,
        prediction_loss_only=True,
        
        # 性能优化
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0, # mulimodal 下多线程可能出问题
        report_to="none",
        # not setting max_steps
    )

    print(f"✅ Training args: epochs={training_args.num_train_epochs}, "
          f"batch_size={training_args.per_device_train_batch_size}")
    
    # 6. 创建 trainer
    collator = MultimodalDPOCollator(processor, max_frames=60)

    # diagnose_trainer_issue(dpo_dataset,collator, processor, peft_model)

    # 🔑 关键诊断：检查可训练参数
    print("🔍 Trainable parameters check:")
    trainable_params = 0
    total_params = 0
    
    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            #print(f"  ✅ Trainable: {name} ({param.numel()} params)")
        # else:
            # print(f"  ❌ Frozen: {name}")
    
    print(f"📊 Total params: {total_params:,}")
    print(f"📊 Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    if trainable_params == 0:
        raise ValueError("❌ No trainable parameters found!")
    
    # 检查是否有语言模型相关的 LoRA 参数
    has_lora_lm = any("language_model" in name and "lora" in name for name, param in peft_model.named_parameters() if param.requires_grad)
    print(f"✅ Language model LoRA parameters: {'Found' if has_lora_lm else 'Missing'}")


    
    try:
        test_batch = collator([dpo_dataset[0]])
        print(f"\n  Collator output type: {type(test_batch)}")
        print(f"  Collator keys: {list(test_batch.keys())}")
        
        # 关键：检查 batch 是否包含 DPOTrainer 需要的字段
        required_keys = ["chosen_input_ids", "rejected_input_ids"]
        for key in required_keys:
            if key in test_batch:
                tensor = test_batch[key]
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    print(f"    ✅ {key}: {tensor.shape} (valid)")
                else:
                    print(f"    ❌ {key}: {tensor} (invalid)")
            else:
                print(f"    ❌ Missing {key}")
                
    except Exception as e:
        print(f"  ❌ Collator failed: {e}")
        import traceback
        traceback.print_exc()
        return


    # 在创建 trainer 前添加：
    print(f"🔍 Final dataset validation:")
    print(f"  Dataset type: {type(dpo_dataset)}")
    print(f"  Dataset features: {dpo_dataset.features if hasattr(dpo_dataset, 'features') else 'N/A'}")

    # 检查是否是 IterableDataset（不应该）
    from datasets import IterableDataset
    if isinstance(dpo_dataset, IterableDataset):
        print("❌ Dataset is IterableDataset! Converting to regular Dataset...")
        dpo_dataset = dpo_dataset.take(len(dpo_dataset))  # 转换为常规数据集

    # 强制转换为 Arrow format
    try:
        dpo_dataset = dpo_dataset.with_format("torch")
        print("✅ Dataset converted to torch format")
    except Exception as e:
        print(f"⚠️ Format conversion failed: {e}")

    # trainer = trainer_cls(
    #     model=peft_model,
    #     args=training_args,
    #     train_dataset=dpo_dataset,
    #     data_collator=collator,
    #     tokenizer=processor.tokenizer,
    #     **trainer_kwargs
    # )

    from training.trainer import create_multimodal_dpo_trainer

    trainer = create_multimodal_dpo_trainer(
        model=peft_model,
        processor=processor,
        train_dataset=dpo_dataset,
        beta=0.1,
        diversity_weight=0.3,
        output_dir="./output/dpo",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
    )

    # 🔑 关键测试：手动获取 dataloader
    print("🔍 Testing trainer dataloader...")
    try:
        train_dataloader = trainer.get_train_dataloader()
        print(f"  Dataloader type: {type(train_dataloader)}")
        print(f"  Dataloader length: {len(train_dataloader) if hasattr(train_dataloader, '__len__') else 'N/A'}")
        
        # 测试第一个 batch
        for batch in train_dataloader:
            print(f"  First batch keys: {list(batch.keys())}")
            print(f"  chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
            print(f"  rejected_input_ids shape: {batch['rejected_input_ids'].shape}")
            break
            
    except Exception as e:
        print(f"❌ Trainer dataloader failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 开始训练
    print(f"🎯 Training {exp_name} with {len(dpo_dataset)} samples...")
    trainer.train()
    
    # 8. 保存模型
    final_path = save_model_path

    merged_model = peft_model.merge_and_unload()
    merged_model,save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"✅ Experiment {exp_na me} completed! Saved to: {final_path}")
    
    return final_path




# 使用方式：
# trainer = diagnose_trainer_issue(dpo_dataset, collator, processor, peft_model)

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal DPO Training")
    parser.add_argument("--rollout_path", type=str, default=f"{rollout_path}/rollout_with_trajectory_k20_n3.json",
                       help="Path to rollout log JSON file")
    parser.add_argument("--output_dir", type=str, default=f"{save_model_path}",
                       help="Output directory")
    parser.add_argument("--experiment", choices=["all", "baseline", "diversity", "adaptive"],
                       default="baseline", help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--diversity_weight", type=float, default=0.3,
                       help="Diversity loss weight")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}


    
    # Baseline DPO 实验
    if args.experiment in ["all", "baseline"]:
        results["baseline"] = run_expe/home/cx/Desktop/UAV/OpenFly/train_t2rl_lora/data_v2/t2rl_train_k20_n3_with_translated.jsonriment(
            exp_name="baseline_dpo",
            #trainer_cls=MultimodalDPOTrainer,
            rollout_path=args.rollout_path,
            output_dir=args.output_dir,
            epochs=args.epochs
        )
    
    # Diversity DPO 实验
    if args.experiment in ["all", "diversity"]:
        results["diversity"] = run_experiment(
            exp_name="diversity_dpo",
            #trainer_cls=EnhancedDiversityDPOTrainer,
            rollout_path=args.rollout_path,
            output_dir=args.output_dir,
            diversity_weight=args.diversity_weight,
            diversity_type="cosine",
            epochs=args.epochs
        )
    
    # Adaptive DPO 实验
    if args.experiment in ["all", "adaptive"]:
        results["adaptive"] = run_experiment(
            exp_name="adaptive_dpo",
            #trainer_cls=AdaptiveDPOTrainer,
            rollout_path=args.rollout_path,
            output_dir=args.output_dir,
            diversity_weight=args.diversity_weight,
            epochs=args.epochs
        )
    
    # 保存实验结果
    results_path = os.path.join(save_model_path, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("📊 EXPERIMENT RESULTS")
    print("="*50)
    for exp, path in results.items():
        print(f"  ✅ {exp.upper()}: {path}")
    print(f"\n📁 Results saved to: {results_path}")
    print("="*50)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    main()