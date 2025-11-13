# 📁 label_generator.py
# (此代码与上一个回答中提供的一致，用于 LLM 具身指令标注)

import json
import os
import argparse
from openai import OpenAI
import random
from typing import Dict, List

# --- 配置 ---
DATA_ROOT = "aerosight_data"
# ⚠️ 请确保您的 OPENAI_API_KEY 环境变量已设置
try:
    LLM_CLIENT = OpenAI() 
except:
    print("Warning: OpenAI client not initialized. Set OPENAI_API_KEY for labeling.")
    LLM_CLIENT = None
    
MODEL_NAME = "gpt-4-turbo"

# LLM 核心 Prompt (结合 OpenFly VLN 规范)
INSTRUCTION_GENERATION_PROMPT = """
你是一名高级的视觉-语言导航（VLN）指令生成专家。你的任务是根据无人机的精确轨迹信息、姿态信息以及沿途的语义观测，逆向推导出一条自然语言指令。这条指令必须能够让 VLA 模型在同样的场景中，精确地复现这条轨迹。

---
**VLN 指令格式规范 (OpenFly 标准):**
1.  **内容格式：** 必须由 **地标 (Landmark)** + **动作序列 (Action Sequence)** 构成。
2.  **长度约束：** 限制在 **1 到 3 个** 连续的短程动作步骤。使用 'Then', 'Next', 'Finally' 等连接词串联步骤。
3.  **参照方式：** 必须包含当前视野内物体的 **相对位置** 和 **视觉属性**。地标必须使用颜色、大小、形状、窗口类型等视觉属性详细描述。

**输入信息 (来自轨迹元数据和语义分析):**
- 轨迹起点和终点的精确 NED 坐标。
- 沿途采集到的关键语义地标和视觉属性（这些信息代表了 VLA 模型的视觉观测）。
- 轨迹执行的动作摘要（方向和距离）。

**任务：** 基于以上信息，生成一条符合 VLN 规范的指令。

**示例（必读）：**
- Trajectory: [Start: Tree-A, End: Building-B] -> [Action: Forward, Yaw Right, Forward]
- Semantic Clues: Tall Green Tree (A), Large Brown Building (B) with Rectangular Windows.
- **Output Instruction:** "Proceed in a straight line toward the large brown building with rectangular windows . Then , slightly turn right and advance forward to the large brown building with rectangular windows . Finally , continue straight to the gray office building with horizontal window blinds ."

---
**请严格遵循规范，仅输出一条指令，不要包含任何 Thought 或其他解释。**
"""

def generate_vln_instruction(trajectory_context: Dict[str, Any]) -> str:
    """调用 LLM API 生成 VLN 指令。"""
    if LLM_CLIENT is None:
        return "LLM_ERROR: Client not ready. Check API key."

    start_pos = trajectory_context['start_pose']['position']
    end_pos = trajectory_context['end_pose']['position']
    
    start_pos_str = f"Start NED: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})"
    end_pos_str = f"End NED: ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f})"
    
    # ⚠️ 实际需要实现一个函数来分析沿途的语义分割图和关键帧，提取地标描述。
    # 这里我们使用一个包含扰动的随机占位符，模拟语义分析的结果。
    mock_semantic_clues = random.choice([
        "Identified a large red warehouse and a tall yellow crane near the start point.",
        "Passed several small green trees and a white circular tank.",
        "The path involved a left turn near a small blue container and ended near a building with blue glass windows."
    ])
    
    user_message = f"""
    - Trajectory ID: {trajectory_context['traj_id']}
    - Pose: {start_pos_str} to {end_pos_str}
    - Key Semantic Clues Found Along Path (VLM/LMM Analysis): {mock_semantic_clues}
    - Action Sequence Summary (Based on Chunk Analysis): [Forward 15m, Slightly Yaw Left, Advance 5m]
    """
    
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": INSTRUCTION_GENERATION_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7 
        )
        # 确保输出是干净的指令
        return response.choices[0].message.content.strip().replace('\n', ' ')
    except Exception as e:
        return f"LLM_ERROR: Failed to generate instruction: {e}"

def main():
    # ... (与上一个回答中的 main 函数保持一致)
    parser = argparse.ArgumentParser(description="AeroSight LLM-based VLN Label Generator.")
    parser.add_argument("--scene_name", type=str, required=True, help="Name of the AirSim scene to process.")
    args = parser.parse_args()
    
    scene_path = os.path.join(DATA_ROOT, args.scene_name)
    metadata_path = os.path.join(scene_path, "trajectories_metadata.jsonl")

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}. Please run data_collector.py first.")
        return

    labeled_data = []
    has_unlabeled = False
    
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        meta = json.loads(line)
        if meta.get("is_labeled") != "SUCCESS":
            has_unlabeled = True
            print(f"-> Generating instruction for {meta['traj_id']}...")
            instruction = generate_vln_instruction(meta)
            
            meta["vln_instruction"] = instruction
            meta["is_labeled"] = "SUCCESS" if not instruction.startswith("LLM_ERROR") else "FAILED"
            print(f"   [Result]: {instruction[:80]}...")
        
        labeled_data.append(meta)

    if not has_unlabeled:
        print("All trajectories already labeled. Skipping generation.")

    # 保存更新后的元数据文件
    with open(metadata_path, 'w') as f:
        for meta in labeled_data:
            f.write(json.dumps(meta) + '\n')
    
    print("\nLabel generation finished. Check the updated metadata.jsonl file.")

if __name__ == "__main__":
    main()