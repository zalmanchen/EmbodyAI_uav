import json
import time
from typing import List, Dict, Any

# --- 导入核心组件 ---
from airsim_client import AirSimClient 
from uav_tools.flight_controls import fly_to_gps, move_forward, set_yaw
from uav_tools.vision_bridge import capture_and_analyze_rgb
from llm_agent_core.memory_manager import MemoryManager
from llm_agent_core.prompt_templates import CORE_SYSTEM_PROMPT, TOOL_SCHEMAS 

# --- 导入 OpenFly VLA 模拟工具 ---
# 假设这是 OpenFly VLA 模型的执行接口
from uav_tools.flight_controls import execute_vln_instruction 

# 📁 main_agent.py (关键修改部分)

# ... 导入语句 ...

# --- 导入 flight_controls 的 CLIENT BINDER ---
from uav_tools.flight_controls import set_airsim_client # <--- 新导入

# --- 初始化全局工具和客户端 ---

# 1. 客户端和连接 
AIRSIM_CLIENT = AirSimClient(vehicle_name="Drone1")
if not AIRSIM_CLIENT.connect_and_initialize():
    print("FATAL ERROR: AirSim 连接失败，程序退出。")
    exit()

# ************ 新增的关键步骤 ************
# 将 AirSimClient 实例绑定到 flight_controls 模块
set_airsim_client(AIRSIM_CLIENT) 
# ****************************************


# 2. 记忆管理器
MEMORY_MANAGER = MemoryManager()

# 3. 可用的工具函数映射
# 注意：我们现在将 execute_vln_instruction 视为 LLM 可直接调用的工具
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
    "execute_vln_instruction": execute_vln_instruction,  # <--- VLA 执行工具
    "report_finding": lambda coords, desc: f"REPORT: 发现目标于 {coords}。详情: {desc}"
}

# --- 模拟 LLM 接口 (双层 Agent 模拟) ---

def mock_llm_call(messages: List[Dict[str, Any]], schemas: Dict[str, Any]) -> Dict[str, Any]:
    """
    【模拟函数】替代真实的 LLM API 调用。
    模拟 LLM 作为高级规划者，决定是进行宏观飞行还是启动 VLA 导航。
    """
    # 获取对话历史中最新的观察结果
    last_observation = messages[-1]['content'] if messages[-1]['role'] != 'system' else ""
    
    # 获取 LLM 的当前状态
    step = len(messages) // 2 # 粗略估计步骤数 (系统消息 + N个(LLM行动 + 工具观察))
    
    # --- 模拟高级规划逻辑 ---
    
    # 1. 初始阶段：起飞
    if "未知" in messages[0]['content'] and "takeoff" not in last_observation:
        print("MOCK LLM: 初始阶段，决定起飞...")
        return {
            "role": "assistant",
            "thought": "任务开始，首先需要起飞到安全高度 20 米。",
            "tool_calls": [{"id": "mock_id_1", "function": {"name": "takeoff", "arguments": '{"altitude": 20.0}'}}]
        }
    
    # 2. 起飞后的第一步：获取姿态并规划宏观飞行
    elif "起飞成功" in last_observation and "get_current_pose" not in [m.get('tool_calls', [{}])[0].get('function', {}).get('name') for m in messages]:
        print("MOCK LLM: 已起飞，决定获取当前位置...")
        return {
            "role": "assistant",
            "thought": "起飞完成，下一步是获取当前精确位置，用于规划前往目标区域的宏观路径。",
            "tool_calls": [{"id": "mock_id_2", "function": {"name": "get_current_pose", "arguments": "{}"}}]
        }
        
    # 3. 宏观飞行到搜索区域
    elif last_observation.startswith("OBSERVATION: 当前姿态") and step < 3:
        print("MOCK LLM: 已知位置，规划宏观飞行到目标区域...")
        # 假设目标区域在 (47.6418, -122.14)
        return {
            "role": "assistant",
            "thought": "已获取当前位置，现在规划飞往搜索区域 (47.6418, -122.14) 的上空 30 米。",
            "tool_calls": [{"id": "mock_id_3", "function": {"name": "fly_to_gps", "arguments": '{"latitude": 47.6418, "longitude": -122.14, "altitude_meters": 30.0}'}}]
        }
    
    # 4. 到达区域后：启动 VLA 低级执行
    elif "成功飞抵目标坐标" in last_observation:
        print("MOCK LLM: 已到达宏观搜索区域，启动 OpenFly VLA 进行精细搜索...")
        # LLM 现在不直接调用 Vision 工具，而是调用 VLA 执行器
        vln_instruction = "在周围寻找红色的背包，并靠近到 3 米以内。"
        return {
            "role": "assistant",
            "thought": f"宏观飞行完成，现在将控制权交给 VLA 模型，执行精细的视觉导航指令：'{vln_instruction}'。",
            "tool_calls": [{"id": "mock_id_4", "function": {"name": "execute_vln_instruction", "arguments": json.dumps({"language_instruction": vln_instruction})}}]
        }
    
    # 5. VLA 执行完成：检查结果并报告
    elif "VLA 模型已完成导航指令" in last_observation:
        print("MOCK LLM: VLA 任务完成，模拟目标发现，决定报告并降落...")
        coords = "Lat: 47.6417, Lon: -122.1401, Alt: 27.0m" # 假设 VLA 找到了更精确的位置
        return {
            "role": "assistant",
            "thought": f"VLA 报告已完成精细搜索，目标已确认在 {coords}。立即报告并返回降落。",
            "tool_calls": [
                {"id": "mock_id_5a", "function": {"name": "report_finding", "arguments": json.dumps({"coordinates": coords, "description": "高度可信的红色目标物，疑似失踪者背包 (VLA 确认)"})}},
                {"id": "mock_id_5b", "function": {"name": "land", "arguments": "{}"}}
            ]
        }
    
    # 6. 默认：未发现/流程结束
    else:
        return {"role": "assistant", "content": "任务流程演示完毕，请手动重置或提供新的指令。"}


# --- 主循环驱动器 (保持不变，但使用更新后的 Prompt) ---

def run_agent(initial_goal: str):
    """驱动 LLM-Agent 的主循环：思考-行动-观察。"""
    
    # 1. 初始化对话历史和系统 Prompt (使用模块化模板)
    system_prompt = CORE_SYSTEM_PROMPT.format(
        initial_goal=initial_goal,
        current_pose="未知"
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
            messages.append(llm_response) 
            
            tool_outputs = []
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                try:
                    function_args = json.loads(tool_call["function"]["arguments"])
                    print(f"   -> 调用工具: {function_name} ({function_args})")
                    
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
            
            messages.extend(tool_outputs) 
        
        # 6. LLM 给出最终报告 (结束任务)
        elif llm_response.get("content"):
            print(f"\n--- 任务结束 ---")
            print(f"Agent 最终报告: {llm_response['content']}")
            break
            
        step_count += 1
        time.sleep(1) 

    if step_count >= max_steps:
        print("\n达到最大执行步数限制，任务终止。")
    
    # 最终清理 (如果在循环中没有降落)
    AIRSIM_CLIENT.land()

if __name__ == "__main__":
    # 运行 Agent 任务
    run_agent("去坐标 (47.6418, -122.14) 附近的区域，寻找红色的标记物或失踪者。")
