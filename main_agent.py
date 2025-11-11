import json
import time
from typing import List, Dict, Any

# --- 导入核心组件 ---
from uav_tools.airsim_client import AirSimClient 
from uav_tools.flight_controls import fly_to_gps, move_forward, set_yaw, move_with_local_avoidance
from uav_tools.vision_bridge import capture_and_analyze_rgb
from uav_tools.vision_bridge import set_airsim_client as set_vision_client # <-- 新导入
from llm_agent_core.memory_manager import MemoryManager
from llm_agent_core.prompt_templates import CORE_SYSTEM_PROMPT, TOOL_SCHEMAS, get_openai_tool_schemas

# --- 导入 OpenFly VLA 模拟工具 ---
# 假设这是 OpenFly VLA 模型的执行接口
from uav_tools.flight_controls import execute_vln_instruction 
from uav_tools.flight_controls import set_airsim_client # <--- 新导入

import os
import openai # <--- 新增导入

import json
import argparse
import os

# --- 全局变量（需要在使用前定义或初始化） ---
AIRSIM_CLIENT = None
MEMORY_MANAGER = None
AVAILABLE_TOOLS = {}


# --- 全局客户端和工具初始化 (封装到函数中) ---

def initialize_agent_system(scene_name: str, initial_goal: str, load_static_map: bool): # <-- add new parameters
    """初始化 AirSim 客户端、记忆管理器和所有工具函数。"""
    
    global AIRSIM_CLIENT, MEMORY_MANAGER, AVAILABLE_TOOLS
    
    # 1. 初始化 AirSim 客户端
    AIRSIM_CLIENT = AirSimClient(vehicle_name="Drone1", env_name=scene_name)
    if not AIRSIM_CLIENT.connect_and_initialize():
        print("FATAL ERROR: AirSim 连接失败，程序退出。")
        return False
        
    # 2. 将客户端实例绑定到 flight_controls 模块
    set_airsim_client(AIRSIM_CLIENT) 

    # 3. 初始化 Memory Manager (使用场景 ID)
    MEMORY_MANAGER = MemoryManager(scene_name=scene_name, load_static_map=load_static_map)
    
    # 4. 检查并导入场景的静态语义数据
    # 路径构建逻辑已封装在 MemoryManager.check_and_initialize_scene_data() 内部
    init_observation = MEMORY_MANAGER.check_and_initialize_scene_data()
    print(f"\n--- 场景初始化观察结果 ---\n{init_observation}\n----------------------------")

    # 5. 可用的工具函数映射 (需要在 global 作用域或此处完整定义)
    # 必须在这里完成定义，以确保它们引用了初始化后的 MEMORY_MANAGER
    global AVAILABLE_TOOLS
    AVAILABLE_TOOLS = {
        "takeoff": AIRSIM_CLIENT.takeoff,
        "land": AIRSIM_CLIENT.land,
        "get_current_pose": AIRSIM_CLIENT.get_current_pose,
        # ... (其他飞行/视觉工具) ...
        # 确保这些引用了新的 MemoryManager 实例
        "fly_to_gps": fly_to_gps, # airsim "inter API, cannot move with avoidance",
        "move_with_local_avoidance": move_with_local_avoidance,
        "move_forward": move_forward,
        "set_yaw": set_yaw,
        "capture_and_analyze_rgb": capture_and_analyze_rgb,
        "update_search_map": MEMORY_MANAGER.update_search_map,
        "retrieve_historical_clues": MEMORY_MANAGER.retrieve_historical_clues,
        "execute_vln_instruction": execute_vln_instruction,  # <--- VLA 执行工具
        "report_finding": lambda coords, desc: f"REPORT: 发现目标于 {coords}。详情: {desc}"
        # ... (OpenFly VLA 工具) ...
    }

    
    return True


# --- 初始化 OpenAI 客户端 ---
try:
    # 客户端将自动查找 OPENAI_API_KEY 环境变量
    OPENAI_CLIENT = openai.OpenAI(
        api_key = "sk-35abc48ab10f4bf1b5b9d9e61f570f5e",   # os.getenv("DASHSCOPE_API_KEY"),
        base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    LLM_MODEL = "qwen3-vl-flash"  # qwen, gpt
except Exception as e:
    print(f"❌ OpenAI 客户端初始化失败: {e}")
    print("请确保已安装 openai 库并设置 OPENAI_API_KEY 环境变量。")
    exit()


# # 2. 记忆管理器
# MEMORY_MANAGER = MemoryManager()


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


def openai_llm_call(messages: List[Dict[str, Any]], available_tools: Dict[str, Any]) -> Dict[str, Any]:
    """
    【真实接口】调用 OpenAI API，启用 Function Calling 功能。
    
    参数:
        messages (List[Dict]): 对话历史。
        available_tools (Dict): 可用的工具函数映射。
        
    返回:
        Dict: LLM 的响应，包含内容或工具调用。
    """
    
    # 1. 获取 OpenAI 格式的工具 Schema
    openai_tool_schemas = get_openai_tool_schemas()
    
    try:
        # 2. 调用 OpenAI API
        response = OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=openai_tool_schemas,
            tool_choice="auto", # 允许模型决定是否调用工具
        )
        
        response_message = response.choices[0].message
        result = {"role": response_message.role}
        
        # 3. 解析 LLM 响应
        
        # 如果 LLM 调用了工具 (Function Call)
        if response_message.tool_calls:
            tool_calls = []
            for call in response_message.tool_calls:
                # 注意：OpenAI 响应中没有 'thought' 字段，所以我们无法直接获取思考过程。
                # 实际应用中，您需要在 LLM 的系统 Prompt 中要求它在每次回复前输出 <thought> 标签。
                tool_calls.append({
                    "id": call.id,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments # JSON 字符串
                    }
                })
            result["tool_calls"] = tool_calls
            
        # 如果 LLM 输出了自然语言内容 (任务完成或询问)
        elif response_message.content:
            # 尝试从内容中提取思考过程 (如果系统提示要求)
            thought = "" # 简略处理，如果需要可以加入正则表达式解析 <thought> 标签
            
            result["content"] = response_message.content
            result["thought"] = thought

        return result
        
    except openai.APIError as e:
        print(f"❌ OpenAI API 错误: {e}")
        return {"role": "assistant", "content": "API 调用失败，任务终止。"}
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return {"role": "assistant", "content": "程序执行错误，任务终止。"}

# --- 主循环驱动器 (保持不变，但使用更新后的 Prompt) ---

def run_agent(initial_goal: str):
    """驱动 LLM-Agent 的主循环：思考-行动-观察。"""
    
    # 1. 初始化对话历史和系统 Prompt
    # a. 构建系统 Prompt
    system_prompt = CORE_SYSTEM_PROMPT.format(
        initial_goal=initial_goal,
        current_pose="未知 (必须先调用 get_current_pose)"
    )
    
    # b. 构造消息列表
    messages = [
        # 必须是列表的第一个元素 (或之一)
        {"role": "system", "content": system_prompt}, 
        
        # ✅ 关键修正：添加用户指令，驱动第一次响应
        {"role": "user", "content": f"请开始执行任务。任务目标是：{initial_goal}"} 
    ]
    
    print("="*60)
    print(f"Agent 主循环启动 | 初始目标: {initial_goal}")
    print("="*60)
    
    # ... (后续循环逻辑保持不变)
    
    max_steps = 10
    step_count = 0
    
    while step_count < max_steps:
        print(f"\n--- 步骤 {step_count + 1}：规划阶段 ---")
        
        # 2. LLM 规划 & 行动 (Function Call)
        llm_response = openai_llm_call(messages, AVAILABLE_TOOLS) # <--- 实际 API 调用
       
        print(llm_response)
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
    parser = argparse.ArgumentParser(description="LLM-Agent 驱动的无人机仿真平台。")
    parser.add_argument(
        "--scene_name", 
        type=str, 
        default= "env_airsim_16",
        help="指定当前的场景名称（例如: env_sim_16），用于记忆隔离和加载seg_map数据。"
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="寻找周围10m最高的楼",
        help="本次任务的初始自然语言目标。"
    )
    # 新增命令行参数：控制是否加载静态地图数据
    parser.add_argument(
        "--load_static_map",
        action="store_true", # 如果命令行提供了此参数，则为 True，否则为 False
        default=False,
        help="如果设置，则从 /scene_data/seg_map 中导入静态语义分割地图信息。"
    )

    args = parser.parse_args()
    
    # 1. 初始化系统
    print(f"=== 启动场景: {args.scene_name} ===")
    if initialize_agent_system(args.scene_name, args.goal, args.load_static_map):
        # 2. 运行 Agent
        run_agent(args.goal)