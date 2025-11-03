import json
import time
from typing import List, Dict, Any

# --- 导入核心组件 ---
from uav_tools.flight_controls import fly_to_gps, move_forward, set_yaw
from uav_tools.vision_bridge import capture_and_analyze_rgb
from llm_agent_core.memory_manager import MemoryManager
from uav_tools.airsim_client import AirSimClient 

# --- 初始化工具和客户端 ---

# 1. 客户端和连接（在 Agent 启动前初始化一次）
AIRSIM_CLIENT = AirSimClient(vehicle_name="Drone1")
if not AIRSIM_CLIENT.connect_and_initialize():
    print("FATAL ERROR: AirSim 连接失败，程序退出。")
    exit()

# 2. 记忆管理器
MEMORY_MANAGER = MemoryManager()

# 3. 可用的工具函数映射
AVAILABLE_TOOLS = {
    "takeoff": AIRSIM_CLIENT.takeoff,
    "land": AIRSIM_CLIENT.land,
    "get_current_pose": AIRSIM_CLIENT.get_current_pose,
    "fly_to_gps": fly_to_gps,
    "move_forward": move_forward,
    "set_yaw": set_yaw,
    "capture_and_analyze_rgb": capture_and_analyze_rgb,
    "update_search_map": MEMORY_MANAGER.update_search_map,
    "retrieve_historical_clues": MEMORY_MANAGER.retrieve_historical_clues,
    # 假设还有一个报告工具，这里我们用一个内部函数模拟
    "report_finding": lambda coords, desc: f"REPORT: 发现目标于 {coords}。详情: {desc}"
}

# --- 定义 LLM 的 Prompt 和 Function Schema (简化版) ---

# 核心 Prompt 模板（用于角色设定和任务指导）
CORE_PROMPT_TEMPLATE = """
你是一个专业的“具身搜索与救援无人机指挥官”。你的目标是根据用户指令，通过调用可用工具来规划、执行无人机任务，并报告发现。

---
## 角色和约束
1. 你的唯一行动方式是调用提供的工具函数 (Function Call)。
2. 你必须遵循“思考-行动-观察”的循环。
3. 你的每次回复必须包含一个 'Thought' 来解释你的推理。

## 当前情境和状态
- **[CURRENT_TASK_GOAL]**: 用户指令的核心任务目标。
- **[CURRENT_DRONE_POSE]**: 无人机当前 GPS 坐标和朝向（来自 'get_current_pose' 工具）。
---
"""

# 简化 Function Schema 定义 (在真实项目中，这将是一个详细的 JSON 列表)
# 这里仅列出 Agent 调用的函数名称及其参数，方便模拟解析
TOOL_SCHEMAS = {
    "takeoff": {"altitude": "float"},
    "fly_to_gps": {"latitude": "float", "longitude": "float", "altitude_meters": "float"},
    "capture_and_analyze_rgb": {"target_description": "str"},
    "report_finding": {"coordinates": "str", "description": "str"},
    "get_current_pose": {}
    # ... (其他工具的简化 schema)
}

# --- 模拟 LLM 接口 ---

def mock_llm_call(messages: List[Dict[str, Any]], schemas: Dict[str, Any]) -> Dict[str, Any]:
    """
    【模拟函数】替代真实的 LLM API 调用。
    根据对话历史和可用工具，返回一个 Function Call 或最终报告。
    """
    last_message = messages[-1]['content']
    
    # 1. 初始阶段：起飞
    if len(messages) == 1:
        print("MOCK LLM: 初始阶段，决定起飞...")
        return {
            "role": "assistant",
            "thought": "任务开始，首先需要起飞到安全高度 20 米。",
            "tool_calls": [{"id": "mock_id_1", "function": {"name": "takeoff", "arguments": '{"altitude": 20.0}'}}]
        }
    
    # 2. 起飞后的第一步：获取姿态并规划飞行
    elif "起飞成功" in last_message and "CURRENT_DRONE_POSE" not in CORE_PROMPT_TEMPLATE:
        print("MOCK LLM: 已起飞，决定获取当前位置...")
        return {
            "role": "assistant",
            "thought": "起飞完成，下一步是获取当前精确位置，用于规划前往目标区域的路径。",
            "tool_calls": [{"id": "mock_id_2", "function": {"name": "get_current_pose", "arguments": "{}"}}]
        }
        
    # 3. 规划前往搜索区域 (使用您 AirSim 环境中的有效 GPS 坐标)
    elif "latitude" in last_message and "longitude" in last_message:
        print("MOCK LLM: 已知位置，规划前往搜索区域...")
        # 假设目标区域在 (47.6418, -122.14)
        return {
            "role": "assistant",
            "thought": "已获取当前位置，现在规划飞往搜索区域 (47.6418, -122.14) 的上空 30 米。",
            "tool_calls": [{"id": "mock_id_3", "function": {"name": "fly_to_gps", "arguments": '{"latitude": 47.6418, "longitude": -122.14, "altitude_meters": 30.0}'}}]
        }
    
    # 4. 到达区域后：开始感知
    elif "成功飞抵目标坐标" in last_message:
        print("MOCK LLM: 已到达目标区域，开始感知...")
        return {
            "role": "assistant",
            "thought": "已到达搜索区域，开始使用视觉工具 'capture_and_analyze_rgb' 寻找目标。",
            "tool_calls": [{"id": "mock_id_4", "function": {"name": "capture_and_analyze_rgb", "arguments": '{"target_description": "寻找红色的标记物或失踪者"}'}}]
        }
    
    # 5. 模拟发现目标并报告
    elif "target_found" in last_message:
        print("MOCK LLM: 发现目标，决定报告并降落...")
        coords = "Lat: 47.6418, Lon: -122.14, Alt: 28.5m"
        return {
            "role": "assistant",
            "thought": f"视觉工具发现目标！坐标 {coords}。我必须立即报告发现，然后降落等待指令。",
            "tool_calls": [
                {"id": "mock_id_5a", "function": {"name": "report_finding", "arguments": json.dumps({"coordinates": coords, "description": "高度可信的红色目标物，疑似失踪者背包"})}},
                {"id": "mock_id_5b", "function": {"name": "land", "arguments": "{}"}}
            ]
        }
    
    # 6. 默认：未发现，继续搜索/结束
    else:
        return {"role": "assistant", "content": "任务流程演示完毕，请手动重置或提供新的指令。"}


# --- 主循环驱动器 ---

def run_agent(initial_goal: str):
    """驱动 LLM-Agent 的主循环：思考-行动-观察。"""
    
    # 1. 初始化对话历史和系统 Prompt
    system_prompt = CORE_PROMPT_TEMPLATE.format(
        CURRENT_TASK_GOAL=initial_goal,
        CURRENT_DRONE_POSE="未知"
    )
    messages = [{"role": "system", "content": system_prompt}]
    
    print("="*60)
    print(f"Agent 主循环启动 | 初始目标: {initial_goal}")
    print("="*60)
    
    max_steps = 10
    step_count = 0
    
    while step_count < max_steps:
        print(f"\n--- 步骤 {step_count + 1}：规划阶段 ---")
        
        # 2. LLM 规划 & 行动 (Function Call)
        llm_response = mock_llm_call(messages, TOOL_SCHEMAS)
        
        # 输出 LLM 的思考过程
        if llm_response.get("thought"):
            print(f"✅ Agent 思考: {llm_response['thought']}")
        
        # 3. 执行工具
        if llm_response.get("tool_calls"):
            tool_calls = llm_response["tool_calls"]
            messages.append(llm_response) # 将 LLM 的行动规划加入对话历史
            
            tool_outputs = []
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                try:
                    # 解析参数，并调用对应的实际 Python 函数
                    function_args = json.loads(tool_call["function"]["arguments"])
                    print(f"   -> 调用工具: {function_name} ({function_args})")
                    
                    # 实际调用工具函数
                    function_to_call = AVAILABLE_TOOLS.get(function_name)
                    if function_to_call:
                        # 4. 执行 & 观察
                        observation = function_to_call(**function_args)
                    else:
                        observation = f"ERROR: 未找到工具函数 {function_name}"
                        
                except Exception as e:
                    observation = f"EXECUTION ERROR: 工具 {function_name} 执行失败: {e}"
                
                print(f"   <- 观察结果 (Observation): {observation[:100]}...")
                
                # 5. 将观察结果反馈给 LLM
                tool_outputs.append({
                    "role": "tool", 
                    "tool_call_id": tool_call["id"], 
                    "content": observation
                })
            
            messages.extend(tool_outputs) # 将所有观察结果加入对话历史
        
        # 6. LLM 给出最终报告 (结束任务)
        elif llm_response.get("content"):
            print(f"\n--- 任务结束 ---")
            print(f"Agent 最终报告: {llm_response['content']}")
            break
            
        step_count += 1
        time.sleep(1) # 模拟处理时间

    if step_count >= max_steps:
        print("\n达到最大执行步数限制，任务终止。")
    
    # 最终清理
    AIRSIM_CLIENT.land()

if __name__ == "__main__":
    # 确保 AirSim 正在运行，并且场景已加载
    run_agent("去坐标 (47.6418, -122.14) 附近的区域，寻找红色的标记物或失踪者。")