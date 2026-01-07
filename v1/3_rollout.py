# 3_rollout.py —— 复用您的 AirSim 逻辑，仅修改输入
import json
import os
import argparse
from scripts.paths import OUTPUT_DIR

OUTPUT_DIR = "/home/cx/Desktop/UAV/OpenFly/train_t2rl_lora/data"

def compute_reward(success, spl, end_dist, traj_len):
    r_succ = 1.0 if success else 0.0
    r_spl = spl if success else 0.0
    r_geo = math.exp(-0.1 * end_dist / max(traj_len, 1e-3))
    return 0.5 * r_succ + 0.3 * r_spl + 0.2 * r_geo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)  # e.g., logs_train_v0.json
    args = parser.parse_args()
    
    # 加载日志
    with open(args.log) as f:
        logs = json.load(f)
    
    # 模拟 AirSim 执行（替换为您的实际 eval）
    updated_logs = []
    for item in logs:
        # 【关键】此处调用您的 AirSim 评估逻辑
        # success, spl, end_dist, traj_len = run_airsim(item)
        
        # 为演示，使用 mock reward
        mock_sr = 0.25 + 0.05 * (hash(item["id"]) % 10) / 10  # 0.25~0.30
        reward = mock_sr * 0.5 + 0.3 * 0.28 + 0.2 * 0.85
        
        item["reward"] = reward
        item["success"] = 1 if mock_sr > 0.28 else 0
        updated_logs.append(item)
    
    # 保存带 reward 的日志
    output_path = args.log.replace(".json", "_with_reward.json")
    with open(output_path, 'w') as f:
        json.dump(updated_logs, f, indent=2)
    print(f"✅ Rollout results saved to {output_path}")

if __name__ == "__main__":
    main()