from transformers import DataCollatorWithPadding
import torch

class MultimodalDPODataCollator(DataCollatorWithPadding):
    """多模态 DPO 数据整理器"""
    
    def __init__(self, processor, max_images=8, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.max_images = max_images
    
    def __call__(self, features):
        # 分离图像路径和文本数据
        batch_images = []
        batch_prompts = []
        batch_chosen = []
        batch_rejected = []
        
        for feature in features:
            # 📌 处理图像路径
            image_paths = feature["images"][:self.max_images]  # 采样
            batch_images.append(image_paths)
            
            batch_prompts.append(feature["prompt"])
            batch_chosen.append(feature["chosen"])
            batch_rejected.append(feature["rejected"])
        
        # 📌 构造 messages 并处理
        chosen_messages = []
        rejected_messages = []
        
        for i in range(len(features)):
            # 构造 chosen messages
            chosen_content = []
            for img_path in batch_images[i]:
                abs_path = os.path.abspath(img_path)
                chosen_content.append({"type": "image", "image": f"file://{abs_path}"})
            chosen_content.append({"type": "text", "text": batch_prompts[i]})
            
            chosen_messages.append([
                {"role": "user", "content": chosen_content}
            ])
            
            # 构造 rejected messages（相同图像，不同文本）
            rejected_content = []
            for img_path in batch_images[i]:
                abs_path = os.path.abspath(img_path)
                rejected_content.append({"type": "image", "image": f"file://{abs_path}"})
            rejected_content.append({"type": "text", "text": batch_prompts[i]})
            
            rejected_messages.append([
                {"role": "user", "content": rejected_content}
            ])
        
        # 📌 使用 process_vision_info 处理
        from qwen_vl_utils import process_vision_info
        
        # 处理 chosen
        chosen_image_inputs = []
        chosen_video_inputs = []
        for msg in chosen_messages:
            img_inp, vid_inp = process_vision_info(msg)
            chosen_image_inputs.extend(img_inp)
            chosen_video_inputs.extend(vid_inp)
        
        # 处理 rejected  
        rejected_image_inputs = []
        rejected_video_inputs = []
        for msg in rejected_messages:
            img_inp, vid_inp = process_vision_info(msg)
            rejected_image_inputs.extend(img_inp)
            rejected_video_inputs.extend(vid_inp)
        
        # 📌 构造文本输入（使用 apply_chat_template）
        chosen_texts = []
        rejected_texts = []
        
        for i in range(len(features)):
            # chosen
            chosen_text = self.processor.apply_chat_template(
                chosen_messages[i], 
                tokenize=False, 
                add_generation_prompt=True
            )
            chosen_texts.append(chosen_text)
            
            # rejected
            rejected_text = self.processor.apply_chat_template(
                rejected_messages[i], 
                tokenize=False, 
                add_generation_prompt=True
            )
            rejected_texts.append(rejected_text)
        
        # 📌 批量处理
        chosen_inputs = self.processor(
            text=chosen_texts,
            images=chosen_image_inputs or None,
            videos=chosen_video_inputs or None,
            padding=True,
            return_tensors="pt"
        )
        
        rejected_inputs = self.processor(
            text=rejected_texts,
            images=rejected_image_inputs or None,
            videos=rejected_video_inputs or None,
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
        }

from trl import DPOTrainer
import torch

class MultimodalDPOTrainer(DPOTrainer):
    """支持多模态模型的 DPOTrainer"""
    
    def get_batch_samples(self, dataloader, num_batches, device=None):
        """重写以支持多模态输入"""
        from transformers.trainer_utils import EvalLoopOutput
        
        # 临时禁用生成评估（多模态模型不支持）
        print("⚠️ Skipping batch samples generation for multimodal model")
        return None, None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """确保 inputs 包含所有必需字段"""
        # 过滤掉 None 的视觉参数
        filtered_inputs = {}
        for k, v in inputs.items():
            if v is not None:
                filtered_inputs[k] = v
        
        return super().compute_loss(model, filtered_inputs, return_outputs)
    
    def training_step(self, model, inputs):
        """处理多模态 inputs"""
        # 确保模型在正确设备上
        model.train()
        
        # 过滤 None 参数
        filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, filtered_inputs)
        
        del filtered_inputs
        return loss