# select_k_train_samples.py
def select_k_samples(train_ list, k: int = 5) -> list:
    """从 train set 中选择 K 条高信息量样本"""
    # 按 reward_gap 排序（需先有 rollout 数据）
    # 或按指令多样性（颜色/动作/建筑类型覆盖）
    selected = []
    seen_weakened = set()
    
    # 示例：优先选不同目标建筑的指令
    building_types = ["beige building", "gray skyscraper", "rooftop"]
    for building in building_types:
        for item in train_
            if building in item["weaken_instruction"] and len(selected) < k:
                if item["weaken_instruction"] not in seen_weakened:
                    selected.append(item)
                    seen_weakened.add(item["weaken_instruction"])
    
    # 补齐至 K 条
    while len(selected) < k and len(train_data) > len(selected):
        candidate = random.choice(train_data)
        if candidate["weaken_instruction"] not in seen_weakened:
            selected.append(candidate)
            seen_weakened.add(candidate["weaken_instruction"])
    
    return selected[:k]