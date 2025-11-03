# 📁 llm_agent_core/prompt_templates.py

from typing import Dict, Any, List

# --- I. 核心系统 Prompt ---

CORE_SYSTEM_PROMPT = """
你是一个专业的“具身搜索与救援无人机指挥官”。你的目标是根据用户指令，通过调用可用工具来规划、执行无人机任务，并报告发现。

---
## 角色和约束
1. 你的唯一行动方式是调用提供的工具函数 (Function Call)。
2. 你必须遵循严格的“思考-行动-观察”的循环。
3. 你的每次回复必须包含一个 'Thought' 字段来解释你的推理过程。
4. 在任何 GPS 飞行指令之前，必须调用 'get_current_pose' 来确认起点。

## 当前情境和状态
- **[CURRENT_TASK_GOAL]**: {initial_goal}
- **[CURRENT_DRONE_POSE]**: {current_pose} (上次获取的姿态)
---
"""

# --- II. Function Calling Schema 定义 ---

# 这是一个简化版的 JSON Schema 格式，用于指导 LLM 理解每个工具的用途。
# 在实际的 OpenAI/Gemini 集成中，你需要使用标准的 JSON Schema V7 格式。

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # 基础控制工具 (来自 airsim_client.py)
    "takeoff": {
        "description": "执行无人机起飞操作，用于开始任务。",
        "parameters": {"altitude": "float (目标高度，单位：米)"}
    },
    "land": {
        "description": "安全降落并解除锁定，用于结束或暂停任务。",
        "parameters": {}
    },
    "get_current_pose": {
        "description": "获取无人机当前的 GPS 坐标、高度和姿态，用于规划路径。",
        "parameters": {}
    },

    # 高级飞行工具 (来自 uav_tools/flight_controls.py)
    "fly_to_gps": {
        "description": "飞往指定的全球定位系统 (GPS) 坐标点。",
        "parameters": {
            "latitude": "float (目标纬度)",
            "longitude": "float (目标经度)",
            "altitude_meters": "float (目标高度，单位：米)"
        }
    },
    # ... (其他飞行工具略) ...
    
    # 视觉感知工具 (来自 uav_tools/vision_bridge.py)
    "capture_and_analyze_rgb": {
        "description": "启动视觉系统，捕获图像并分析指定目标（模拟 YOLO 或 VLM）。",
        "parameters": {"target_description": "str (要寻找的目标的自然语言描述，例如：'红色的背包')" }
    },

    # 长期记忆/RAG 工具 (来自 llm_agent_core/memory_manager.py)
    "update_search_map": {
        "description": "Agent 发现新线索或完成搜索区域时调用，用于存储关键记忆。",
        "parameters": {
            "coordinates": "dict (发现点的GPS坐标)",
            "status": "str (记忆状态：'线索', '已搜索', '禁飞区')",
            "description": "str (详细描述)"
        }
    },
    "retrieve_historical_clues": {
        "description": "根据语义查询，从长期记忆中召回最相关的历史线索，用于规划。",
        "parameters": {"query": "str (LLM 的自然语言查询)"}
    },

    # OpenFly VLA 集成工具 (未来关键)
    "execute_vln_instruction": {
        "description": "激活 OpenFly VLA 模型，执行短程、反应式的视觉-语言导航指令。",
        "parameters": {"language_instruction": "str (VLA 要执行的短语指令)"}
    },

    # 报告工具
    "report_finding": {
        "description": "当确定找到任务目标时，必须调用此函数。",
        "parameters": {
            "coordinates": "str (目标的 GPS 坐标)",
            "description": "str (目标的详细描述)"
        }
    }
}