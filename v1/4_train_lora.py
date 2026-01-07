# 4_train_lora.py
import os
import json
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# 配置
from scripts.paths import OUTPUT_DIR

def prepare_dpo_data(log_path):
    with open(log_path) as f:
        logs = json.load(f)
    
    # 按 weakened_instruction 分组
    groups = {}
    for log in logs:
        key = log["weakened_instruction"]
        if key not in groups:
            groups[key] = []
        groups[key].append(log)
    
    # 构建 preference pairs
    dpo_data = []
    for weaken, candidates in groups.items():
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda x: x["reward"], reverse=True)
        dpo_data.append({
            "prompt": f"Human: {weaken}\nAssistant: ",
            "chosen": candidates[0]["translated_instruction"],
            "rejected": candidates[-1]["translated_instruction"]
        })
    return dpo_data

def main():
    # 加载数据（仅 train set reward）
    train_log = os.path.join(OUTPUT_DIR, "logs_train_v0_with_reward.json")
    dpo_data = prepare_dpo_data(train_log)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True
    )
    
    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["wqkv", "wo"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 训练
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            learning_rate=2e-5,
            per_device_train_batch_size=3,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            output_dir=os.path.join(OUTPUT_DIR, "t2rl_lora")
        ),
        train_dataset=dpo_data
    )
    trainer.train()
    
    # 保存
    model.save_pretrained(os.path.join(OUTPUT_DIR, "t2rl_lora"))
    print("✅ LoRA saved!")

if __name__ == "__main__":
    main()