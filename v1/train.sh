# 1. Few-Shot 翻译
# python translate.py --split train --k_shot 5 --n_samples 3

# 2. Rollout（关键！生成 DPO 数据）
python eval.py \
  --log data/logs_train_8xA100_7B_n3_k5_diverse.json \
  --output data/logs_k5_with_reward.json

# 3. 构建偏好对
python build_preference_pairs.py --log data/logs_k5_with_reward.json



python ./train_t2rl_lora/v1/2_translate_v1.py --split train --k_shot 5 --n_samples 3 --output ./train_t2rl_lora/data/logs_train_8xA100_7B_n3_k5_diverse.json

python ./train_t2rl_lora/v1/3_rollout_v1.py --log ./train_t2rl_lora/data/logs_train_8xA100_7B_n3_k5_diverse.json --output ./train_t2rl_lora/data/logs_k5_with_reward.json

python ./train_t2rl_lora/v1/4_train_lora_v3.py --log ./train_t2rl_lora/data/logs_k5_with_reward.json --output ./train_t2rl_lora/output/t2rl_lora_k5_v2 --diversity_weight 0.3 --lr 2e-4

python ./train_t2rl_lora/v1/5_evaluate.py