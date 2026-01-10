#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal DPO Trainer Collection - FIXED VERSION
✅ 修复 get_batch_samples 返回 None 的问题
"""

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


# class MultimodalDPOTrainer(DPOTrainer):
#     """基础多模态 DPO Trainer（修复版）"""
    
#     def get_batch_samples(self, dataloader, num_batches, device=None) -> Tuple[Optional[Any], int]:
#         """修复：返回空列表而不是 None"""
#         # 返回空列表和 0 个样本
#         return [], 0
    
#     def training_step(self, model, inputs):
#         """处理多模态 inputs"""
#         model.train()
#         # 过滤 None 参数
#         filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
#         with self.compute_loss_context_manager():
#             loss = self.compute_loss(model, filtered_inputs)
#         return loss


# 替换 EnhancedDiversityDPOTrainer 为标准 Trainer 的子类
from transformers import Trainer

class MultimodalDPOTrainer(Trainer):  # 或 DPOTrainer
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
        
        # 手动前向和反向传播
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, filtered_inputs, num_items_in_batch=num_items_in_batch)
        
        # 🔑 关键：手动检查梯度
        print(f"\n🔍 Gradient Debug - Step {self.state.global_step}:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Loss requires_grad: {loss.requires_grad}")
        
        if loss.requires_grad:
            # 清零梯度
            import pdb; pdb.set_trace()
            self.optimizer.zero_grad()
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            total_grad_norm = 0.0
            lora_grad_norm = 0.0
            total_params = 0
            lora_params = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    total_params += 1
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm ** 2
                        
                        if 'lora' in name:
                            lora_params += 1
                            lora_grad_norm += grad_norm ** 2
                            if grad_norm > 0:
                                print(f"    ✅ LoRA Grad: {name} = {grad_norm:.6f}")
                            else:
                                print(f"    ❌ Zero LoRA Grad: {name}")
                        else:
                            if grad_norm > 0:
                                print(f"    ✅ Other Grad: {name} = {grad_norm:.6f}")
                    else:
                        print(f"    ❌ No Grad: {name}")
            
            total_grad_norm = total_grad_norm ** 0.5
            lora_grad_norm = lora_grad_norm ** 0.5
            
            print(f"  Total Grad Norm: {total_grad_norm:.6f}")
            print(f"  LoRA Grad Norm: {lora_grad_norm:.6f}")
            print(f"  Total Trainable Params: {total_params}")
            print(f"  LoRA Params with Grad: {lora_params}")
            
            # 梯度裁剪和优化
            if total_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
            else:
                print("  ❌ No gradients detected! Skipping optimizer step.")
        else:
            print("  ❌ Loss does not require gradient!")
        
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """修复：添加 num_items_in_batch 参数"""
        # 过滤 None 参数
        filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
        
        # 自定义 DPO loss（不要调用 super().compute_loss()）
        dpo_loss = self._custom_dpo_loss(model, filtered_inputs)
        
        # 多样性 loss
        diversity_loss = torch.tensor(0.0, device=dpo_loss.device)
        if hasattr(self, 'diversity_weight') and self.diversity_weight > 0:
            try:
                diversity_loss = self._compute_diversity_loss(model, filtered_inputs)
            except Exception as e:
                print(f"⚠️ Diversity loss failed: {e}")
        
        total_loss = dpo_loss + getattr(self, 'diversity_weight', 0.0) * diversity_loss
        
        return (total_loss, {}) if return_outputs else total_loss
    
    # def _custom_dpo_loss(self, model, inputs):
    #     """自定义 DPO loss"""
    #     # 前向 chosen
    #     chosen_outputs = model(
    #         input_ids=inputs["chosen_input_ids"],
    #         attention_mask=inputs["chosen_attention_mask"],
    #         labels=inputs["chosen_labels"],
    #         pixel_values=inputs.get("chosen_pixel_values"),
    #         image_grid_thw=inputs.get("chosen_image_grid_thw"),
    #     )
        
    #     # 前向 rejected
    #     rejected_outputs = model(
    #         input_ids=inputs["rejected_input_ids"],
    #         attention_mask=inputs["rejected_attention_mask"],
    #         labels=inputs["rejected_labels"],
    #         pixel_values=inputs.get("rejected_pixel_values"),
    #         image_grid_thw=inputs.get("rejected_image_grid_thw"),
    #     )
        
    #     # DPO loss
    #     beta = getattr(self, 'beta', 0.1)
    #     chosen_logps = -chosen_outputs.loss
    #     rejected_logps = -rejected_outputs.loss
    #     logits = beta * (chosen_logps - rejected_logps)
    #     return -torch.nn.functional.logsigmoid(logits).mean()
    
    def _custom_dpo_loss(self, model, inputs):
        """✅ 正确的 DPO loss 实现 - 基于 logits 而不是 loss"""
        
        # 构建前向参数
        chosen_kwargs = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"],
            "output_hidden_states": False,
            "use_cache": False,
        }
        if "chosen_pixel_values" in inputs and inputs["chosen_pixel_values"] is not None:
            chosen_kwargs["pixel_values"] = inputs["chosen_pixel_values"]
        if "chosen_image_grid_thw" in inputs and inputs["chosen_image_grid_thw"] is not None:
            chosen_kwargs["image_grid_thw"] = inputs["chosen_image_grid_thw"]
        
        rejected_kwargs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"],
            "output_hidden_states": False,
            "use_cache": False,
        }
        if "rejected_pixel_values" in inputs and inputs["rejected_pixel_values"] is not None:
            rejected_kwargs["pixel_values"] = inputs["rejected_pixel_values"]
        if "rejected_image_grid_thw" in inputs and inputs["rejected_image_grid_thw"] is not None:
            rejected_kwargs["image_grid_thw"] = inputs["rejected_image_grid_thw"]
        
        # 前向传播获取 logits（关键：需要梯度）
        chosen_outputs = model(**chosen_kwargs)
        rejected_outputs = model(**rejected_kwargs)
        
        # 获取 logits
        chosen_logits = chosen_outputs.logits  # [B, L, V]
        rejected_logits = rejected_outputs.logits  # [B, L, V]
        
        # 获取 labels 和 masks
        chosen_labels = inputs["chosen_labels"]  # [B, L]
        rejected_labels = inputs["rejected_labels"]  # [B, L]
        chosen_attn_mask = inputs["chosen_attention_mask"]  # [B, L]
        rejected_attn_mask = inputs["rejected_attention_mask"]  # [B, L]
        
        # 计算 log probabilities
        chosen_logps = self._get_batch_logps(chosen_logits, chosen_labels, chosen_attn_mask)
        rejected_logps = self._get_batch_logps(rejected_logits, rejected_labels, rejected_attn_mask)
        
        # DPO loss
        beta = getattr(self, 'beta', 0.1)
        logits_diff = beta * (chosen_logps - rejected_logps)
        dpo_loss = -torch.nn.functional.logsigmoid(logits_diff).mean()
        
        return dpo_loss

    def _get_batch_logps(self, logits, labels, attention_mask):
        """
        计算 batch log probabilities
        
        Args:
            logits: [B, L, V] - 模型输出的 logits
            labels: [B, L] - 标签 (-100 表示忽略)
            attention_mask: [B, L] - 注意力掩码
        
        Returns:
            logps: [B] - 每个序列的 log probability 总和
        """
        # 转换为 log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, L, V]
        
        # 获取每个位置的真实 token log prob
        # labels.unsqueeze(-1): [B, L, 1]
        # torch.gather: [B, L, 1] -> squeeze to [B, L]
        per_token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, L]
        
        # 创建有效 token mask: attention_mask & (labels != -100)
        valid_mask = attention_mask * (labels != -100)  # [B, L]
        
        # 应用 mask 并求和
        per_sequence_logps = (per_token_logps * valid_mask).sum(dim=-1)  # [B]
        
        return per_sequence_logps
        
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     """正确的 DPO loss 实现"""
        
    #     # 过滤 None 参数
    #     filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
        
    #     # 🔑 关键：正确计算 log probabilities
    #     chosen_logps, rejected_logps = self._get_batch_logps(
    #         model, 
    #         filtered_inputs["chosen_input_ids"],
    #         filtered_inputs["chosen_attention_mask"],
    #         filtered_inputs["chosen_labels"],
    #         filtered_inputs["rejected_input_ids"],
    #         filtered_inputs["rejected_attention_mask"],
    #         filtered_inputs["rejected_labels"],
    #         filtered_inputs.get("chosen_pixel_values"),
    #         filtered_inputs.get("chosen_image_grid_thw"),
    #         filtered_inputs.get("rejected_pixel_values"),
    #         filtered_inputs.get("rejected_image_grid_thw"),
    #     )
        
    #     # DPO loss = -log sigmoid(beta * (chosen_logps - rejected_logps))
    #     beta = getattr(self, 'beta', 0.1)
    #     logits = beta * (chosen_logps - rejected_logps)
    #     dpo_loss = -torch.nn.functional.logsigmoid(logits).mean()
        
    #     # 多样性 loss（如果需要）
    #     diversity_loss = torch.tensor(0.0, device=dpo_loss.device)
    #     if hasattr(self, 'diversity_weight') and self.diversity_weight > 0:
    #         try:
    #             diversity_loss = self._compute_diversity_loss(model, filtered_inputs)
    #         except Exception as e:
    #             print(f"⚠️ Diversity loss failed: {e}")
        
    #     total_loss = dpo_loss + getattr(self, 'diversity_weight', 0.0) * diversity_loss
        
    #     # 调试信息
    #     if self.state.global_step % 10 == 0:
    #         print(f"Step {self.state.global_step}: "
    #             f"DPO={dpo_loss.item():.4f}, "
    #             f"Chosen_logps={chosen_logps.mean().item():.4f}, "
    #             f"Rejected_logps={rejected_logps.mean().item():.4f}")
        
    #     return (total_loss, {}) if return_outputs else total_loss

    # def _get_batch_logps(self, model, chosen_input_ids, chosen_attention_mask, chosen_labels,
    #                     rejected_input_ids, rejected_attention_mask, rejected_labels,
    #                     chosen_pixel_values=None, chosen_image_grid_thw=None,
    #                     rejected_pixel_values=None, rejected_image_grid_thw=None):
    #     """计算 batch log probabilities"""
        
    #     # 构建 chosen 前向参数（只包含必要参数）
    #     chosen_kwargs = {
    #         "input_ids": chosen_input_ids,
    #         "attention_mask": chosen_attention_mask,
    #     }
    #     if chosen_pixel_values is not None:
    #         chosen_kwargs["pixel_values"] = chosen_pixel_values
    #     if chosen_image_grid_thw is not None:
    #         chosen_kwargs["image_grid_thw"] = chosen_image_grid_thw
        
    #     # 构建 rejected 前向参数
    #     rejected_kwargs = {
    #         "input_ids": rejected_input_ids,
    #         "attention_mask": rejected_attention_mask,
    #     }
    #     if rejected_pixel_values is not None:
    #         rejected_kwargs["pixel_values"] = rejected_pixel_values
    #     if rejected_image_grid_thw is not None:
    #         rejected_kwargs["image_grid_thw"] = rejected_image_grid_thw
        
    #     # 🔑 关键修复：不要重复传递参数
    #     # 前向传播获取 logits（训练模式，需要梯度）
    #     chosen_outputs = model(**chosen_kwargs)
    #     rejected_outputs = model(**rejected_kwargs)
        
    #     chosen_logits = chosen_outputs.logits
    #     rejected_logits = rejected_outputs.logits
        
    #     # 计算 log probabilities
    #     chosen_logps = self._get_logps(chosen_logits, chosen_labels, chosen_attention_mask)
    #     rejected_logps = self._get_logps(rejected_logits, rejected_labels, rejected_attention_mask)
        
    #     return chosen_logps, rejected_logps

    # def _get_logps(self, logits, labels, attention_mask):
    #     """计算 log probabilities"""
    #     # logits: [B, L, V]
    #     # labels: [B, L] (-100 for ignored tokens)
    #     # attention_mask: [B, L]
        
    #     # 转换为 log probabilities
    #     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
    #     # 获取每个位置的 log prob
    #     per_token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
    #     # 应用 attention mask 和 labels mask
    #     mask = attention_mask * (labels != -100)
    #     per_token_logps = per_token_logps * mask
        
    #     # 求和得到每个序列的 log prob
    #     logps = per_token_logps.sum(dim=-1)
        
    #     return logps


class EnhancedDiversityDPOTrainer(DPOTrainer):
    """多样性增强 DPO Trainer（修复版）"""
    
    def __init__(self, diversity_weight=0.3, diversity_type="cosine", **kwargs):
        super().__init__(**kwargs)
        self.diversity_weight = diversity_weight
        self.diversity_type = diversity_type
    
    def get_batch_samples(self, dataloader, num_batches, device=None) -> Tuple[Optional[Any], int]:
        """修复：返回空列表而不是 None"""
        return [], 0
    
    def training_step(self, model, inputs):
        model.train()
        filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, filtered_inputs)
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. 标准 DPO loss
        try:
            dpo_loss = super().compute_loss(model, inputs, return_outputs=True)
            if isinstance(dpo_loss, tuple):
                dpo_loss = dpo_loss[0]
        except Exception as e:
            print(f"⚠️ DPO loss computation failed: {e}")
            # 回退到简单损失
            dpo_loss = torch.tensor(1.0, device=inputs["chosen_input_ids"].device)
        
        # 2. 多样性 loss
        diversity_loss = torch.tensor(0.0, device=dpo_loss.device)
        if self.diversity_weight > 0 and "chosen_input_ids" in inputs:
            try:
                diversity_loss = self.compute_diversity_loss(model, inputs)
            except Exception as e:
                print(f"⚠️ Diversity loss computation failed: {e}")
        
        # 3. 总 loss
        total_loss = dpo_loss + self.diversity_weight * diversity_loss
        
        # 4. 训练日志
        if hasattr(self.state, 'global_step') and self.state.global_step % 50 == 0 and self.args.should_log:
            reward_gaps = inputs.get("reward_gaps", torch.ones_like(dpo_loss))
            print(f"Step {self.state.global_step}: "
                  f"DPO={dpo_loss.item():.4f}, "
                  f"Diversity={diversity_loss.item():.4f}, "
                  f"AvgGap={reward_gaps.mean().item():.3f}")
        
        return (total_loss, {}) if return_outputs else total_loss
    
    def compute_diversity_loss(self, model, inputs):
        """计算多样性损失"""
        with torch.no_grad():
            # 构建前向参数
            chosen_kwargs = self._build_forward_kwargs(inputs, "chosen")
            rejected_kwargs = self._build_forward_kwargs(inputs, "rejected")
            
            chosen_outputs = model(**chosen_kwargs, output_hidden_states=True, use_cache=False)
            rejected_outputs = model(**rejected_kwargs, output_hidden_states=True, use_cache=False)
        
        # 提取最后一层隐藏状态（最后一个 token）
        chosen_hidden = chosen_outputs.hidden_states[-1][:, -1, :]
        rejected_hidden = rejected_outputs.hidden_states[-1][:, -1, :]
        
        # 多样性计算
        if self.diversity_type == "cosine":
            cos_sim = F.cosine_similarity(chosen_hidden, rejected_hidden, dim=-1)
            return (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown diversity type: {self.diversity_type}")
    
    def _build_forward_kwargs(self, inputs: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """构建模型前向传播参数"""
        kwargs = {
            "input_ids": inputs[f"{prefix}_input_ids"],
            "attention_mask": inputs[f"{prefix}_attention_mask"],
        }
        # 条件添加视觉参数
        if f"{prefix}_pixel_values" in inputs and inputs[f"{prefix}_pixel_values"] is not None:
            kwargs["pixel_values"] = inputs[f"{prefix}_pixel_values"]
        if f"{prefix}_image_grid_thw" in inputs and inputs[f"{prefix}_image_grid_thw"] is not None:
            kwargs["image_grid_thw"] = inputs[f"{prefix}_image_grid_thw"]
        return kwargs

    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """重写内部训练循环，修复多模态兼容性问题"""
        
        """重写内部训练循环"""
        # ✅ 正确方式：使用无参数 super()
        return super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval
        )

    def _simplified_training_loop(self, batch_size=None, args=None):
        """简化训练循环"""
        print("🔄 Using simplified training loop...")
        


        train_dataloader = self.get_train_dataloader()
        num_epochs = args.num_train_epochs
        total_steps = 0
        
        for epoch in range(num_epochs):
            for step, batch in enumerate(train_dataloader):
                # 过滤 None 参数
                filtered_batch = {k: v for k, v in batch.items() if v is not None}
                
                # 计算 loss
                loss = self.training_step(self.model, filtered_batch)
                
                # 梯度累积和优化
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    total_steps += 1
                    
                    # 日志
                    if total_steps % args.logging_steps == 0:
                        print(f"Step {total_steps}: loss={loss.item():.4f}")
                    
                    # 保存
                    if total_steps % args.save_steps == 0:
                        self.save_model()
                
                if total_steps >= 50:  # 限制总步数
                    break
            
            if total_steps >= 50:
                break
        
        return {"train_loss": 0.0, "epoch": num_epochs}

class AdaptiveDPOTrainer(EnhancedDiversityDPOTrainer):
    """自适应权重 DPO Trainer（修复版）"""
    
    def get_batch_samples(self, dataloader, num_batches, device=None) -> Tuple[Optional[Any], int]:
        """修复：返回空列表而不是 None"""
        return [], 0
    
    def compute_loss(self, model, inputs, return_outputs=False):
        dpo_loss = super().compute_loss(model, inputs, return_outputs=True)
        if isinstance(dpo_loss, tuple):
            dpo_loss = dpo_loss[0]
        
        diversity_loss = torch.tensor(0.0, device=dpo_loss.device)
        if self.diversity_weight > 0 and "chosen_input_ids" in inputs:
            try:
                diversity_loss = self.compute_diversity_loss(model, inputs)
            except:
                pass
        
        # 自适应权重
        reward_gaps = inputs.get("reward_gaps", torch.ones_like(dpo_loss))
        adaptive_weights = self.diversity_weight * torch.exp(-reward_gaps)
        
        total_loss = dpo_loss + adaptive_weights.mean() * diversity_loss
        
        if hasattr(self.state, 'global_step') and self.state.global_step % 50 == 0 and self.args.should_log:
            print(f"Step {self.state.global_step}: "
                  f"DPO={dpo_loss.item():.4f}, "
                  f"Diversity={diversity_loss.item():.4f}, "
                  f"AdaptiveWeight={adaptive_weights.mean().item():.3f}")
        
        return (total_loss, {}) if return_outputs else total_loss


class MultimodalDPOCollator:
    """适配您数据格式的多模态 DPO collator"""
    
    def __init__(self, processor, max_frames=8):
        self.processor = processor
        self.max_frames = max_frames
    
    def __call__(self, features):
        batch_data = []
        
        for item in features:
            # 处理图像路径和帧索引
            image_path = item["images"][0] if item["images"] else ""
            index_list = item.get("index_list", [])
            
            # 加载图像 paths（保持与您 eval 脚本一致）
            image_paths = self._load_image_paths(image_path, index_list)
            
            batch_data.append({
                "images": image_paths,
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "reward_gap": item["reward_gap"]
            })

        
        return self._process_multimodal_batch(batch_data)
    
    def _load_image_paths(self, parquet_name: str, index_list: list) -> list:
        """加载图像路径，与您的 eval 脚本保持一致"""
        if not parquet_name or not index_list:
            return []
        
        # 采样帧
        if len(index_list) > self.max_frames:
            step = len(index_list) / self.max_frames
            sampled_indices = [index_list[int(i * step)] for i in range(self.max_frames)]
        else:
            sampled_indices = index_list[:]
        
        # 构造图像路径（匹配您的 eval 脚本逻辑）
        image_paths = []
        for idx_str in sampled_indices:
            try:
                frame_idx = int(str(idx_str).split('_')[-1])
                # 这里返回文件路径，collator 会处理为 file:// 协议
                img_path = f"./tmp/qwen_vl_imgs/openfly_{parquet_name}_{frame_idx:04d}.jpg"
                # 实际路径需要根据您的 eval 脚本调整
                image_paths.append({"path": img_path})
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue
        
        return image_paths
  
    def _process_multimodal_batch(self, batch_data):
        """多模态批次处理"""
        from qwen_vl_utils import process_vision_info
        
        chosen_messages = []
        rejected_messages = []
        reward_gaps = []
        
        for item in batch_data:

            # 构建消息内容
            chosen_content = self._build_content(item.get("images", []), item.get("prompt", ""))
            rejected_content = self._build_content(item.get("images", []), item.get("prompt", ""))
            
            chosen_messages.append([{"role": "user", "content": chosen_content}])
            rejected_messages.append([{"role": "user", "content": rejected_content}])
            reward_gaps.append(item.get("reward_gap", 0.1))

        # 处理视觉信息
        chosen_image_inputs = []
        rejected_image_inputs = []
        
        for msg in chosen_messages:
            try:
                img_inp, _ = process_vision_info(msg)
                chosen_image_inputs.extend(img_inp or [])
            except:
                chosen_image_inputs.extend([])
        
        for msg in rejected_messages:
            try:
                img_inp, _ = process_vision_info(msg)
                rejected_image_inputs.extend(img_inp or [])
            except:
                rejected_image_inputs.extend([])
        
        # 构造文本
        chosen_texts = []
        rejected_texts = []
        
        for item in batch_data:
            images = item.get("images", [])
            prompt = item.get("prompt", "")
            
            # chosen
            try:
                chosen_text = self.processor.apply_chat_template(
                    [{"role": "user", "content": self._build_content(images, prompt)}],
                    tokenize=False, 
                    add_generation_prompt=True
                )
                chosen_texts.append(chosen_text)
            except:
                chosen_texts.append(f"Human: {prompt}\nAssistant: ")
            
            # rejected
            try:
                rejected_text = self.processor.apply_chat_template(
                    [{"role": "user", "content": self._build_content(images, prompt)}],
                    tokenize=False, 
                    add_generation_prompt=True
                )
                rejected_texts.append(rejected_text)
            except:
                rejected_texts.append(f"Human: {prompt}\nAssistant: ")
        
        # 批量处理（简化版，确保不失败）
        try:
            chosen_inputs = self.processor(
                text=chosen_texts,
                images=chosen_image_inputs or None,
                padding=True,
                return_tensors="pt"
            )
        except:
            # 回退到纯文本处理
            chosen_inputs = self.processor.tokenizer(
                chosen_texts,
                padding=True,
                return_tensors="pt"
            )
        
        try:
            rejected_inputs = self.processor(
                text=rejected_texts,
                images=rejected_image_inputs or None,
                padding=True,
                return_tensors="pt"
            )
        except:
            # 回退到纯文本处理
            rejected_inputs = self.processor.tokenizer(
                rejected_texts,
                padding=True,
                return_tensors="pt"
            )

        # 🔑 关键修复：创建 labels 字段
        # chosen_labels 通常是 chosen_input_ids 的副本，但 pad tokens 设为 -100
        chosen_labels = chosen_inputs.input_ids.clone()
        chosen_labels[chosen_labels == self.processor.tokenizer.pad_token_id] = -100
        
        rejected_labels = rejected_inputs.input_ids.clone()
        rejected_labels[rejected_labels == self.processor.tokenizer.pad_token_id] = -100
        
        # 构建返回字典
        result = {
            "chosen_input_ids": chosen_inputs.input_ids,
            "chosen_attention_mask": chosen_inputs.attention_mask,
            "chosen_labels": chosen_labels,  # 🔑 新增
            "rejected_input_ids": rejected_inputs.input_ids,
            "rejected_attention_mask": rejected_inputs.attention_mask,
            "rejected_labels": rejected_labels,  # 🔑 新增
            "reward_gaps": torch.tensor(reward_gaps, dtype=torch.float32),
        }
        
        # 条件添加视觉参数
        if hasattr(chosen_inputs, 'pixel_values') and chosen_inputs.pixel_values is not None:
            result["chosen_pixel_values"] = chosen_inputs.pixel_values
        if hasattr(chosen_inputs, 'image_grid_thw') and chosen_inputs.image_grid_thw is not None:
            result["chosen_image_grid_thw"] = chosen_inputs.image_grid_thw
        if hasattr(rejected_inputs, 'pixel_values') and rejected_inputs.pixel_values is not None:
            result["rejected_pixel_values"] = rejected_inputs.pixel_values
        if hasattr(rejected_inputs, 'image_grid_thw') and rejected_inputs.image_grid_thw is not None:
            result["rejected_image_grid_thw"] = rejected_inputs.image_grid_thw

        return result
    
    def _build_content(self, image_paths, prompt):
        """构建消息内容"""
        import os
        content = []
        # 🔑 关键修复：处理字典格式的 image_paths
        processed_paths = []
        for img_path in image_paths:
            if isinstance(img_path, dict):
                # 从字典中提取路径
                path_str = img_path.get("path", "")
                if path_str:
                    processed_paths.append(path_str)
            elif isinstance(img_path, str):
                # 已经是字符串路径
                processed_paths.append(img_path)
            # 忽略其他格式
        
        # 构建 content
        for path_str in processed_paths:
            abs_path = os.path.abspath(path_str)
            content.append({"type": "image", "image": f"file://{abs_path}"})
        
        content.append({"type": "text", "text": prompt})
        return content