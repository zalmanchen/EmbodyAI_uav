import os
import safetensors.torch

lora_path = "./train_t2rl_lora/output/t2rl_lora_k5_v3"
weights = safetensors.torch.load_file(
    os.path.join(lora_path, "adapter_model.safetensors")
)

print("🔍 LoRA weight keys (first 5):")
for i, k in enumerate(list(weights.keys())[:5]):
    print(f"  {i}: {k}")

# 检查 base model state_dict 键名
from transformers import AutoModelForVision2Seq
base_model = AutoModelForVision2Seq.from_pretrained(
    "./model/qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True,
    device_map="cpu"
)
print("\n🔍 Base model state_dict keys (first 5):")
for i, k in enumerate(list(base_model.state_dict().keys())[:5]):
    print(f"  {i}: {k}")