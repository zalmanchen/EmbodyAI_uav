
from trl import DPOTrainer
import torch.nn.functional as F

class MultimodalDiversityDPOCollator:
    """多模态 + 多样性友好的 DPO 数据整理器"""
    
    def __init__(self, processor, max_images=8, diversity_level="token"):
        self.processor = processor
        self.max_images = max_images
        self.diversity_level = diversity_level  # "token", "sequence", "semantic"
    
    def __call__(self, features):
        # 分组：相同 prompt 的样本归为一组（用于多样性计算）
        prompt_groups = {}
        for feature in features:
            prompt = feature["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(feature)
        
        # 为每组构造多样化的样本对
        batch_data = []
        for prompt, group in prompt_groups.items():
            if len(group) >= 2:
                # 构造多个 (chosen, rejected) 对以增强多样性
                for i in range(len(group)):
                    for j in range(len(group)):
                        if i != j and group[i]["reward"] > group[j]["reward"] + 0.1:
                            batch_data.append({
                                "images": group[i]["images"],
                                "prompt": prompt,
                                "chosen": group[i]["text"],
                                "rejected": group[j]["text"],
                                "reward_gap": group[i]["reward"] - group[j]["reward"],
                                "diversity_type": "inter_sample"  # 样本间多样性
                            })
        
        # 如果样本不足，回退到标准 DPO 对
        if len(batch_data) < len(features):
            batch_data = features[:len(batch_data)] if batch_data else features
        
        # 调用基础多模态处理
        return self._process_multimodal_batch(batch_data)
    
    def _process_multimodal_batch(self, batch_data):
        """基础多模态处理逻辑"""
        from qwen_vl_utils import process_vision_info
        
        chosen_messages = []
        rejected_messages = []
        diversity_labels = []
        
        for item in batch_data:
            # 处理 chosen
            chosen_content = self._build_content(item["images"], item["prompt"])
            chosen_messages.append([{"role": "user", "content": chosen_content}])
            
            # 处理 rejected  
            rejected_content = self._build_content(item["images"], item["prompt"])
            rejected_messages.append([{"role": "user", "content": rejected_content}])
            
            diversity_labels.append(item.get("diversity_type", "standard"))
        
        # 使用 process_vision_info 处理
        chosen_image_inputs = []
        rejected_image_inputs = []
        
        for msg in chosen_messages:
            img_inp, _ = process_vision_info(msg)
            chosen_image_inputs.extend(img_inp)
        
        for msg in rejected_messages:
            img_inp, _ = process_vision_info(msg)
            rejected_image_inputs.extend(img_inp)
        
        # 构造文本
        chosen_texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": self._build_content(item["images"], item["prompt"])}],
                tokenize=False, 
                add_generation_prompt=True
            )
            for item in batch_data
        ]
        
        rejected_texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": self._build_content(item["images"], item["prompt"])}],
                tokenize=False, 
                add_generation_prompt=True
            )
            for item in batch_data
        ]
        
        # 批量处理
        chosen_inputs = self.processor(
            text=chosen_texts,
            images=chosen_image_inputs or None,
            padding=True,
            return_tensors="pt"
        )
        
        rejected_inputs = self.processor(
            text=rejected_texts,
            images=rejected_image_inputs or None,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_inputs.input_ids,
            "chosen_attention_mask": chosen_inputs.attention_mask,
            "chosen_pixel_values": chosen_inputs.get("pixel_values"),
            "chosen_image_grid_thw": chosen_inputs.get("image_grid_thw"),
            "rejected_input_ids": rejected_inputs.input_ids,
            "rejected_attention_mask": rejected_inputs.attention_mask,
            "rejected_pixel_values": rejected_inputs.get("pixel_values"),
            "rejected_image_grid_thw": rejected_inputs.get("image_grid_thw"),
            "diversity_labels": diversity_labels  # 用于多样性损失计算
        }
    
    def _build_content(self, image_paths, prompt):
        """构建消息内容"""
        content = []
        for img_path in image_paths[:self.max_images]:
            abs_path = os.path.abspath(img_path)
            content.append({"type": "image", "image": f"file://{abs_path}"})
        content.append({"type": "text", "text": prompt})
        return content


class EnhancedDiversityDPOTrainer(DPOTrainer):
    """支持多模态输入的多样性 DPO trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. 标准 DPO loss
        dpo_loss = super().compute_loss(model, inputs, return_outputs=True)
        
        # 2. 🔑 多样性损失（多模态感知）
        diversity_loss = self.compute_enhanced_diversity_loss(model, inputs)
        
        # 3. 自适应权重（基于 reward gap）
        reward_gaps = inputs.get("reward_gaps", torch.ones_like(dpo_loss))
        adaptive_weight = self.diversity_weight * torch.clamp(reward_gaps, 0.1, 1.0)
        
        total_loss = dpo_loss + adaptive_weight.mean() * diversity_loss
        
        if self.state.global_step % 100 == 0:
            print(f"Step {self.state.global_step}: DPO={dpo_loss.item():.4f}, "
                  f"Diversity={diversity_loss.item():.4f}, "
                  f"Weight={adaptive_weight.mean().item():.2f}")
        
        return (total_loss, {}) if return_outputs else total_loss
    
    def compute_enhanced_diversity_loss(self, model, inputs):
        """增强的多样性损失计算"""
        with torch.no_grad():
            # 获取 chosen 和 rejected 的最后一层隐藏状态
            chosen_outputs = model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
                pixel_values=inputs.get("chosen_pixel_values"),
                image_grid_thw=inputs.get("chosen_image_grid_thw"),
                output_hidden_states=True,
                use_cache=False
            )
            
            rejected_outputs = model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
                pixel_values=inputs.get("rejected_pixel_values"),
                image_grid_thw=inputs.get("rejected_image_grid_thw"),
                output_hidden_states=True,
                use_cache=False
            )
        
        # 提取最后一层隐藏状态（排除图像 tokens）
        chosen_hidden = chosen_outputs.hidden_states[-1][:, -1, :]  # 最后一个 token
        rejected_hidden = rejected_outputs.hidden_states[-1][:, -1, :]
        
        # 多样性损失：鼓励 chosen 和 rejected 差异化
        cos_sim = F.cosine_similarity(chosen_hidden, rejected_hidden, dim=-1)
        diversity_loss = (1 - cos_sim).mean()  # 1.0 = 完全不同, 0.0 = 完全相同
        
        return diversity_loss