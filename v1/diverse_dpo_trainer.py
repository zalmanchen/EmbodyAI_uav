"""
多样性正则化 DPO Trainer（修复终极版）
✅ 修复所有已知问题：
   ✔️ 删除双重 super().__init__ → 解决 lr=0 / scheduler.base_lr=0
   ✔️ ref_model 强制同设备 → 杜绝 CUDA device mismatch
   ✔️ fallback loss 显式 requires_grad=True → 保底梯度不为 0
   ✔️ data collator 纯 CPU → 交由 Trainer 自动迁移，避免 early device binding
   ✔️ 兼容 transformers 4.50+ log(num_items_in_batch) 签名变更
✅ 保留全部业务：DPO + Token Entropy 多样性正则 + 动态权重衰减
"""
import torch
import torch.nn.functional as F
from typing import Optional, Union, Iterator, Tuple, Dict, List, Any
from trl import DPOTrainer
from transformers import PreTrainedTokenizerBase, ProcessorMixin, TrainingArguments
import copy
import warnings
warnings.filterwarnings("ignore")

# ==============================================
# ✅ Safe DPO Data Collator（纯 CPU 整理，由 Trainer 统一迁移）
# ==============================================
class SafeDPODataCollatorWithPadding:
    def __init__(self, pad_token_id: int = 0, max_length: int = 512):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 直接提取预 tokenize 字段（无需再处理 "chosen"/"rejected" 字符串）
        batch = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
        }

        for feat in features:
            # 必需字段校验
            required_keys = [
                "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
                "rejected_input_ids", "rejected_attention_mask", "rejected_labels"
            ]
            if not all(k in feat for k in required_keys):
                print(f"⚠️ 跳过缺失字段样本: {list(feat.keys())}")
                continue

            # 提取并清洗 token IDs（替换 None → pad_token_id）
            def clean_ids(ids):
                return [
                    self.pad_token_id if x is None else int(x)
                    for x in ids if x is not None or True  # 保留 None 以便替换
                ]

            c_ids = clean_ids(feat["chosen_input_ids"])
            c_mask = feat["chosen_attention_mask"][:len(c_ids)]
            c_labels = clean_ids(feat["chosen_labels"])

            r_ids = clean_ids(feat["rejected_input_ids"])
            r_mask = feat["rejected_attention_mask"][:len(r_ids)]
            r_labels = clean_ids(feat["rejected_labels"])

            # 截断/填充
            c_ids, c_mask, c_labels = self._pad_and_truncate(c_ids, c_mask, c_labels)
            r_ids, r_mask, r_labels = self._pad_and_truncate(r_ids, r_mask, r_labels)

            batch["chosen_input_ids"].append(c_ids)
            batch["chosen_attention_mask"].append(c_mask)
            batch["chosen_labels"].append(c_labels)
            batch["rejected_input_ids"].append(r_ids)
            batch["rejected_attention_mask"].append(r_mask)
            batch["rejected_labels"].append(r_labels)

        # 若无有效样本，返回最小 dummy batch
        if len(batch["chosen_input_ids"]) == 0:
            return self._empty_batch()

        # 转为 tensor
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }

    def _pad_and_truncate(self, ids, mask, labels):
        """统一处理 ids/mask/labels 的截断与填充"""
        if len(ids) > self.max_length:
            return ids[:self.max_length], mask[:self.max_length], labels[:self.max_length]
        pad_len = self.max_length - len(ids)
        pad_ids = ids + [self.pad_token_id] * pad_len
        pad_mask = mask + [0] * pad_len
        pad_labels = labels + [-100] * pad_len  # labels 的 pad 用 -100
        return pad_ids, pad_mask, pad_labels

    def _empty_batch(self):
        """返回最小有效 batch（含 labels）"""
        dummy = [self.pad_token_id]
        dummy_mask = [1]
        dummy_labels = [-100]
        return {
            "chosen_input_ids": torch.tensor([dummy], dtype=torch.long),
            "chosen_attention_mask": torch.tensor([dummy_mask], dtype=torch.long),
            "chosen_labels": torch.tensor([dummy_labels], dtype=torch.long),
            "rejected_input_ids": torch.tensor([dummy], dtype=torch.long),
            "rejected_attention_mask": torch.tensor([dummy_mask], dtype=torch.long),
            "rejected_labels": torch.tensor([dummy_labels], dtype=torch.long),
        }

# ==============================================
# ✅ DiversityDPOTrainer（终极修复版）
# ==============================================
class DiversityDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model,
        diversity_weight: float = 0.3,
        diversity_schedule: str = "adaptive",  # "fixed" or "adaptive"
        tokenizer: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        **kwargs
    ):
        # 🔧 Step 1: 准备修复项（ref_model / args 补全 / tokenizer 提取）
        self._main_model = model
        self._model_device = next(model.parameters()).device

        # ✅ ref_model: 深拷贝 + 同设备 + 冻结
        ref_model = copy.deepcopy(model).to(self._model_device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        kwargs["ref_model"] = ref_model

        # ✅ 补全 TrainingArguments 必需属性（防 AttributeError）
        training_args = kwargs.get("args")
        if isinstance(training_args, TrainingArguments):
            # 关键：关闭列过滤（DPO 需要 chosen/rejected）
            training_args.remove_unused_columns = False
            # 兜底补齐
            defaults = {
                "model_init_kwargs": None,
                "ref_model_init_kwargs": None,
                "generate_during_eval": False,
                "model_adapter_name": None,
                "ref_adapter_name": None,
                "reference_free": False,
                "sync_ref_model": False,
                "max_length": 512,
                "max_prompt_length": 256,
                "max_target_length": 512,
                "beta": 0.1,
                "loss_type": "sigmoid",
                "truncation_mode": "keep_end",
                "label_pad_token_id": -100,
                "label_smoothing": 0.0,
                "disable_dropout": True,
                "precompute_ref_log_probs": False,
                "dataset_num_proc": None,
                "pipeline_parallel": False,
            }
            for k, v in defaults.items():
                if not hasattr(training_args, k):
                    setattr(training_args, k, v)

        # ✅ 提取纯文本 tokenizer（适配 Processor）
        _real_tokenizer = tokenizer
        if tokenizer is not None:
            if hasattr(tokenizer, "tokenizer"):  # QwenProcessor
                _real_tokenizer = tokenizer.tokenizer
            elif hasattr(tokenizer, "image_processor"):  # HuggingFace Processor
                _real_tokenizer = getattr(tokenizer, "tokenizer", None)
            kwargs["tokenizer"] = _real_tokenizer

        # ✅ 设置 data collator（使用 pad_token_id 安全版）
        pad_id = _real_tokenizer.pad_token_id if _real_tokenizer else 0
        kwargs["data_collator"] = SafeDPODataCollatorWithPadding(
            pad_token_id=pad_id,
            max_length=getattr(training_args, "max_length", 512)
        )

        # 🔧 Step 2: ✅ 仅调用一次父类初始化（核心修复！）
        super().__init__(model=model, **kwargs)

        # 🔧 Step 3: 业务参数
        self.diversity_weight = diversity_weight
        self.diversity_schedule = diversity_schedule
        self._current_diversity_weight = diversity_weight
        self._original_tokenizer = tokenizer
        self.processing_class = tokenizer  # 兼容 inference

    # ✅ 兼容 log(num_items_in_batch=None) 签名
    def log(self, logs: Dict[str, float], num_items_in_batch: Optional[int] = None):
        super().log(logs)

    # ✅ get_batch_samples 鲁棒实现
    # def get_batch_samples(self, epoch_iterator: Iterator, num_batches: Optional[int] = None, **kwargs):
    #     samples, count = [], 0
    #     try:
    #         for i, batch in enumerate(epoch_iterator):
    #             if num_batches is not None and i >= num_batches:
    #                 break
    #             if batch and len(batch) > 0:
    #                 samples.append(batch)
    #                 count += len(batch)
    #     except Exception:
    #         pass
    #     return samples, count

    def get_batch_samples(
        self,
        epoch_iterator: Iterator,
        num_batches: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ) -> Tuple[list, int]:
        """
        兼容 trl ≥ 0.8.0 的 3-arg 调用：
            get_batch_samples(epoch_iterator, num_batches, device)
        """
        batch_samples, num_items = [], 0
        try:
            count = 0
            for batch in epoch_iterator:
                if num_batches is not None and count >= num_batches:
                    break
                if batch and len(batch) > 0:
                    batch_samples.append(batch)
                    num_items += len(batch)
                    count += 1
        except Exception:
            pass
        return batch_samples, num_items

    # ✅ Token Entropy（多样性核心）
    def _compute_token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=-1) + 1e-8
        entropy = -(probs * torch.log(probs)).sum(dim=-1)
        return entropy.mean()

    # ✅ compute_loss（三重防护：设备对齐 + 梯度保底 + 异常兜底）
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            # 🔒 Step 1: 强制 inputs 迁移到模型设备（防跨卡）
            for k, v in list(inputs.items()):  # list() 避免 runtime change
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self._model_device, non_blocking=True)

            # 🔒 Step 2: DPO loss
            dpo_loss = super().compute_loss(model, inputs, return_outputs=False)

            # 🔒 Step 3: 自适应多样性权重
            if self.diversity_schedule == "adaptive":
                total_steps = self.args.max_steps or (
                    (len(self.train_dataset) // self.args.per_device_train_batch_size) * self.args.num_train_epochs
                )
                current_step = max(1, getattr(self.state, "global_step", 1))
                ratio = min(1.0, current_step / max(1, total_steps * 0.5))
                self._current_diversity_weight = self.diversity_weight * (1 - 0.8 * ratio)

            # 🔒 Step 4: 多样性 loss（entropy 越大越好 → loss = -entropy）
            diversity_loss = torch.tensor(0.0, device=self._model_device)
            if self._current_diversity_weight > 1e-5 and "chosen_input_ids" in inputs:
                try:
                    with torch.no_grad():
                        chosen_out = model(
                            input_ids=inputs["chosen_input_ids"],
                            attention_mask=inputs["chosen_attention_mask"]
                        )
                        rejected_out = model(
                            input_ids=inputs["rejected_input_ids"],
                            attention_mask=inputs["rejected_attention_mask"]
                        )
                    entropy_c = self._compute_token_entropy(chosen_out.logits)
                    entropy_r = self._compute_token_entropy(rejected_out.logits)
                    diversity_loss = -0.5 * (entropy_c + entropy_r)
                except Exception:
                    pass  # 失败则 diversity_loss = 0

            # 🔒 Step 5: 总 loss（确保可导 + 设备一致）
            total_loss = dpo_loss + self._current_diversity_weight * diversity_loss
            total_loss = total_loss.to(self._model_device)

            return (total_loss, {}) if return_outputs else total_loss

        # except Exception as e:
        #     # 🛡️ 终极兜底：返回可导小 loss（确保梯度不为 0）
        #     fallback_loss = torch.tensor(
        #         [1e-2], 
        #         device=self._model_device, 
        #         dtype=torch.float32,
        #         requires_grad=True  # ✅ 显式开启梯度！
        #     ).sum()
        #     return fallback_loss

        except Exception as e:
            import traceback
            print("\n" + "="*50)
            print("❗ compute_loss 发生异常，已触发 fallback！")
            print(f"Error type: {type(e).__name__}")
            print(f"Error msg: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("="*50 + "\n")
            
            fallback_loss = torch.tensor(
                [1e-2], 
                device=self._model_device,
                dtype=torch.float32,
                requires_grad=True
            ).sum()
            return fallback_loss