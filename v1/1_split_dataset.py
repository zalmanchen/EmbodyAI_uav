# 1_split_dataset.py
import json
import random
import os

# 加载配置
#from scripts.paths import TRAIN_JSON_PATH, EVAL_JSON_PATH, OUTPUT_DIR

TRAIN_JSON_PATH = "/home/cx/Desktop/UAV/data/openfly/train_env_airsim_18_with_weaken.json"
EVAL_JSON_PATH = "/home/cx/Desktop/UAV/data/openfly/env_airsim_18_with_weaken.json"
OUTPUT_DIR = "/home/cx/Desktop/UAV/OpenFly/train_t2rl_lora/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 加载原始数据
    with open(TRAIN_JSON_PATH) as f:
        train_data = json.load(f)
    with open(EVAL_JSON_PATH) as f:
        eval_data = json.load(f)
    
    # 随机划分 Train/Val (85/15)
    random.seed(42)
    random.shuffle(train_data)
    #n_train = int(0.85 * len(train_data))
    n_train = 100
    splits = {
        "train": train_data[:20],
        "val": train_data[20:40],
        "eval": eval_data  # 冻结！
    }
    
    # 保存
    for name, data in splits.items():
        path = os.path.join(OUTPUT_DIR, f"t2rl_{name}_k20.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ {name}: {len(data)} items → {path}")
    
    print("\n🔍 Next: python 2_translate.py --split train")

if __name__ == "__main__":
    main()