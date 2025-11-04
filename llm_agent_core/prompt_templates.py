# 📁 llm_agent_core/prompt_templates.py

from typing import Dict, Any, List

# --- I. 核心系统 Prompt ---

# 📁 llm_agent_core/prompt_templates.py

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
**【强制行动指令】**：
**在你收到任务目标后，第一步必须是调用 'get_current_pose' 来确认无人机状态和位置，准备执行任务。在任务未完成之前，禁止输出任何总结性的或报告性的内容。**
---
"""

# --- II. 内部工具定义（简化格式） ---

# 这是一个内部字典，用于定义工具的名称、描述和参数。
# 注意：这里使用 Python 类型字符串，而不是 JSON Schema 类型。
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # 基础控制工具 (来自 airsim_client.py)
    "takeoff": {
        "description": "执行无人机起飞操作，用于开始任务。起飞到安全高度。",
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
        "description": "飞往指定的全球定位系统 (GPS) 坐标点。这是高级规划层的主要宏观行动工具。",
        "parameters": {
            "latitude": "float (目标纬度)",
            "longitude": "float (目标经度)",
            "altitude_meters": "float (目标高度，单位：米)"
        }
    },
    
    # 视觉感知工具 (来自 uav_tools/vision_bridge.py)
    "capture_and_analyze_rgb": {
        "description": "启动视觉系统，捕获图像并分析指定目标（模拟 YOLO 或 VLM）。",
        "parameters": {"target_description": "str (要寻找的目标的自然语言描述，例如：'红色的背包')" }
    },

    # OpenFly VLA 集成工具
    "execute_vln_instruction": {
        "description": "激活 OpenFly VLA 模型，执行短程、反应式的视觉-语言导航指令。当无人机接近目标区域时使用。",
        "parameters": {"language_instruction": "str (VLA 要执行的短语指令，如'向左边的大树靠近')" }
    },

    # 报告工具
    "report_finding": {
        "description": "当确定找到任务目标时，必须调用此函数。",
        "parameters": {
            "coordinates": "str (目标的 GPS 坐标，格式为 Lat: x.xxxx, Lon: y.yyyy, Alt: z.zzm)",
            "description": "str (目标的详细描述)"
        }
    }
}

# --- III. 转换函数 (解决导入错误) ---

# 类型映射表，用于将 Python 字符串映射到 JSON Schema 类型
TYPE_MAP = {
    "float": "number",
    "str": "string",
    "dict": "object",
    "list": "array",
    "int": "integer"
}


def get_openai_tool_schemas() -> List[Dict[str, Any]]:
    """
    将 TOOL_SCHEMAS 转换为 OpenAI Function Calling 所需的严格 JSON Schema 格式。
    """
    openai_schemas = []
    
    for name, definition in TOOL_SCHEMAS.items():
        function_schema = {
            "name": name,
            "description": definition["description"],
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # 处理参数
        if "parameters" in definition:
            for param_name, type_str in definition["parameters"].items():
                # 简单解析类型和描述
                parts = type_str.split(' ', 1)
                param_type = parts[0].lower()
                param_desc = parts[1] if len(parts) > 1 else f"参数 {param_name}"

                # 转换为 JSON Schema 类型
                json_type = TYPE_MAP.get(param_type, "string") # 默认为 string
                
                function_schema["parameters"]["properties"][param_name] = {
                    "type": json_type,
                    "description": param_desc
                }
                # 假设所有定义的参数都是必填的
                function_schema["parameters"]["required"].append(param_name)

        openai_schemas.append({
            "type": "function",
            "function": function_schema
        })
        
    return openai_schemas