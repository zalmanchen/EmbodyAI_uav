#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal DPO Trainer Collection - FIXED VERSION
✅ 修复 get_batch_samples 返回 None 的问题
"""

import os
import torch
import torch.nn.functional as F
from trl import DPOTrainer
from typing import Dict, Any, Tuple, Optional


def build_system_prompt(agent_name: str = "openfly") -> str:
    role = (
    "You are a precision UAV instruction translator specialized in aerial navigation.\n"
    "Your task is to convert high-level human instructions into detailed, executable flight commands\n"
    "that maintain strict adherence to the observed visual trajectory and flight sequence."
    )

    few_shot = """
    ### Navigation Command Style (Observe Pattern):
    "Head directly toward the tall, light beige building with many windows. Then, slightly turn right and proceed to another large building characterized by its light gray color and balcony-like structures. Finally, slightly turn left and continue straight towards a tall, multi-story skyscraper with large, beige windows featuring arched tops."
    "Proceed directly to the grey urban rooftop featuring antennas and equipment on a medium-sized building. Then, slightly turn left and head straight towards it."
    "Advance towards the gray skyscraper characterized by a tall building. Then, slightly turn right and proceed to it. Finally, slightly turn left and continue straight to it."
    """.strip()

    constraints = (
        "\n### Translation Requirements:\n"
        "• MAINTAIN flight order: preserve the exact sequence of targets\n"
        "• GROUND in visuals: describe only objects and features visible in the trajectory\n"
        "• NO hallucinations: omit objects, colors, or structures not present in frames\n"
        "\n### Output Specifications:\n"
        "- Single coherent paragraph\n"
        "- Complete sentences with proper punctuation\n"
        "- Professional technical language suitable for UAV operations"
    )
    return f"{role}\n\n{few_shot}\n{constraints}"



from trl import DPOTrainer
from transformers import TrainingArguments
# from transformers import DPOConfig

original_get_batch_samples = DPOTrainer.get_batch_samples

def patched_get_batch_samples(self, dataloader, num_batches, device=None):
    """多模态模型不支持生成样本评估"""
    return [], 0

DPOTrainer.get_batch_samples = patched_get_batch_samples

from transformers import TrainingArguments, Trainer, AutoModelForVision2Seq, AutoProcessor

class MultimodalDPOTrainer(Trainer):
    """标准 Trainer + 自定义 DPO loss"""
    
    def __init__(self, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """修复：构建完整的 prompt + response 序列"""
        
        filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
        
        # 👉 关键：拼接 prompt + chosen
        full_chosen_input_ids = torch.cat([
            filtered_inputs["prompt_input_ids"],
            filtered_inputs["chosen_input_ids"]
        ], dim=1)
        
        full_chosen_attention_mask = torch.cat([
            filtered_inputs["prompt_attention_mask"], 
            filtered_inputs["chosen_attention_mask"]
        ], dim=1)
        
        # Labels: prompt 部分 ignore (-100), chosen 部分计算 loss
        full_chosen_labels = torch.cat([
            torch.full_like(filtered_inputs["prompt_input_ids"], -100),
            filtered_inputs["chosen_labels"]
        ], dim=1)
        
        # 同样处理 rejected
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
        
        # 👉 视觉参数保持不变（来自 prompt 处理）
        vision_kwargs = {}
        if "pixel_values" in filtered_inputs:
            vision_kwargs["pixel_values"] = filtered_inputs["pixel_values"]
        if "image_grid_thw" in filtered_inputs:
            vision_kwargs["image_grid_thw"] = filtered_inputs["image_grid_thw"]
        
        # 前向传播
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
        
        # DPO loss 计算
        chosen_logps = -chosen_outputs.loss
        rejected_logps = -rejected_outputs.loss
        logits = self.beta * (chosen_logps - rejected_logps)
        dpo_loss = -F.logsigmoid(logits).mean()
        
        return dpo_loss


class MultimodalDPOCollator:
    """支持 system prompt 的 Qwen2.5-VL 多模态 collator"""
    
    def __init__(self, processor, max_frames=8, system_prompt=None):
        self.processor = processor
        self.max_frames = max_frames
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # 确保 tokenizer 配置正确
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = 151643
        if self.processor.tokenizer.eos_token_id is None:
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
            # 构建完整对话：system + user(images + prompt)
            messages = []
            
            # System message
            messages.append({"role": "system", "content": self.system_prompt})
            
            # User message with images and prompt
            content = []
            image_paths = item["images"][:self.max_frames]
            
            # 添加图像
            for img_path in image_paths:
                if isinstance(img_path, dict):
                    path_str = img_path.get("path", "")
                else:
                    path_str = img_path
                
                if path_str:
                    # 尝试处理不同路径格式
                    if os.path.exists(path_str):
                        abs_path = os.path.abspath(path_str)

                        content.append({"type": "image", "image": f"file://{abs_path}"})
                    else:
                        # 可能是相对路径或其他格式
                        content.append({"type": "image", "image": path_str})
            
            # 添加文本指令
            content.append({"type": "text", "text": item["prompt"]})
            messages.append({"role": "user", "content": content})
            
            prompt_messages.append(messages)
            chosen_responses.append(item["chosen"])
            rejected_responses.append(item["rejected"])
            
            # 处理视觉信息
            try:
                vision_input, _ = process_vision_info(messages)
                if vision_input:
                    all_vision_inputs.extend(vision_input)
            except Exception as e:
                print(f"⚠️ Vision processing failed: {e}")
                continue
        
        # 应用 chat template 获取文本
        prompt_texts = []
        for messages in prompt_messages:
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt_texts.append(text)
            except Exception as e:
                print(f"⚠️ Chat template failed: {e}")
                # 回退：手动构建包含 system prompt 的文本
                num_imgs = len([c for msg in messages for c in (msg.get("content", []) if isinstance(msg.get("content"), list) else [msg.get("content")]) if isinstance(c, dict) and c.get("type") == "image"])
                placeholder = "<tool_call>" * num_imgs + "<tool_call>"
                system_part = f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
                user_part = f"<|im_start|>user\n{placeholder}\n{messages[-1]['content'][-1]['text']}<|im_end|>"
                prompt_texts.append(f"{system_part}\n{user_part}")
        

        prompt_inputs = self.processor(text=prompt_texts,images=all_vision_inputs,padding=True,return_tensors="pt")

        # 处理 prompt + images
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
        
        # 处理 responses (纯文本，包含 assistant role)
        chosen_texts = []
        rejected_texts = []
        for chosen, rejected in zip(chosen_responses, rejected_responses):
            # 添加 assistant role 到 responses
            chosen_texts.append(f"<|im_start|>assistant\n{chosen}<|im_end|>")
            rejected_texts.append(f"<|im_start|>assistant\n{rejected}<|im_end|>")
        
        chosen_inputs = self.processor.tokenizer(
            chosen_texts,
            padding=True,
            return_tensors="pt"
        )
        rejected_inputs = self.processor.tokenizer(
            rejected_texts,
            padding=True,
            return_tensors="pt"
        )
        
        # 创建 labels
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
        
        # 添加视觉参数
        if hasattr(prompt_inputs, 'pixel_values') and prompt_inputs.pixel_values is not None:
            result['pixel_values'] = prompt_inputs.pixel_values
        if hasattr(prompt_inputs, 'image_grid_thw') and prompt_inputs.image_grid_thw is not None:
            result['image_grid_thw'] = prompt_inputs.image_grid_thw
            
        return result



# 创建训练器
def create_multimodal_dpo_trainer(
    model,
    processor,
    train_dataset,
    beta=0.1,
    output_dir="./output/dpo",
    **training_kwargs
):
    """创建符合 DPOTrainer 规范的多模态训练器"""
    
    system_prompt = build_system_prompt()
    collator = MultimodalDPOCollator(processor, max_frames=60, system_prompt=system_prompt)
    

    # 测试 collator 输出
    test_batch = collator([train_dataset[0]])
    print(f"Prompt text sample: {processor.tokenizer.decode(test_batch['prompt_input_ids'][0][:20])}")
    print(f"Pixel values shape: {test_batch.get('pixel_values', 'None')}")
    print(f"Number of <image> tokens: {(test_batch['prompt_input_ids'][0] == processor.tokenizer.encode('<image>')[0]).sum()}")

    batch_size = 1
    grad_acc_steps = 4
    save_limit = 15

    steps_per_epoch = len(train_dataset) // (batch_size * grad_acc_steps)
    if len(train_dataset) % (batch_size * grad_acc_steps) != 0:
        steps_per_epoch += 1

    save_steps = steps_per_epoch * save_limit

    # 4. 训练配置
    training_args = TrainingArguments(
        output_dir="./output/v2",
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=5e-5,
        logging_steps=30,
        save_strategy="epoch",
        save_steps=save_steps,
        save_total_limit=save_limit,
        eval_strategy="no",
        # predict_with_generate=False,
        include_inputs_for_metrics=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,

        # key: add grad norm
        max_grad_norm = 1.0, # limit grad_norm
        warmup_ratio = 0.1 # warmup learning rate
    )
    
    # 5. 创建并训练 DPOTrainer
    trainer = MultimodalDPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    
    

    return trainer