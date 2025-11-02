好的，为您生成项目的完整 **README 文件**，包含项目的概览、代码结构、安装指南、代码逻辑思路以及如何运行原型验证。

-----

# 🚁 LLM-Agent 驱动的无人机搜索与救援仿真平台 (UAV-SAR)

## 📖 项目概览

本项目旨在一个高保真物理仿真环境（基于 **AirSim** 和 **Unreal Engine**）中，构建一个由大型语言模型（LLM）驱动的具身智能体（Embodied Agent）。该 Agent 能够接收自然语言指令，自主规划飞行路径，利用视觉感知系统（**YOLO 模型**）识别目标，并使用 **RAG（检索增强生成）记忆系统** 进行长期的情境推理，执行复杂的搜索与救援（SAR）任务。

## 🧠 核心代码逻辑思路 (Code Logic Breakdown)

本项目采用经典的 **“思考 (Thought) - 行动 (Action) - 观察 (Observation)”** 循环框架，通过 **Function Calling** 机制将 LLM 的认知能力与无人机的物理执行能力桥接起来。

1.  **启动与初始化 (`main_agent.py`)**

      * **连接 AirSim：** 实例化 `AirSimClient`，建立与 AirSim 仿真环境的 RPC 连接，并进行无人机解锁和环境重置。
      * **记忆初始化：** 初始化 `MemoryManager`（基于 ChromaDB 的向量数据库），建立 Agent 的长期记忆。
      * **工具映射：** 将所有 Python 工具函数（飞行、感知、记忆）注册到一个字典中，供主循环调用。

2.  **LLM 驱动循环 (`run_agent` in `main_agent.py`)**

      * **规划 (Thought)：** LLM 接收系统 Prompt、当前状态和历史对话（记忆）。它生成一个 `Thought`，解释其推理过程（例如：需要前往新区域、开始感知、避障等）。
      * **行动 (Action)：** LLM 决定调用一个或多个工具（Function Call），以 JSON 格式输出函数名和参数。
      * **执行 (Execution)：** 主循环解析 LLM 的 JSON 输出，通过 **工具映射** 查找对应的 Python 函数，并执行无人机操作。
      * **观察 (Observation)：** 工具函数执行完毕后，返回一个结构化的 **Observation** 报告（例如：`OBSERVATION: 成功飞抵目标坐标` 或 YOLO 的检测结果 JSON）。
      * **记忆更新：** 如果 Observation 包含新线索或已完成的搜索状态，Agent 调用 `update_search_map` 将其存储到长期记忆中。

3.  **感知系统 (`vision_bridge.py`)**

      * **目标检测：** 在 `capture_and_analyze_rgb` 中，模拟（或实际调用）YOLO 模型对 AirSim 捕获的 RGB 图像进行实时目标检测。
      * **具身感知：** 实现 **像素坐标到世界坐标的转换** 逻辑，将 YOLO 输出的 2D 像素位置，结合 AirSim 的深度图和无人机姿态，计算出目标的精确 3D GPS 坐标。

4.  **记忆与推理 (`memory_manager.py`)**

      * **存储：** `update_search_map` 将 Agent 的重要发现（如线索、已搜索区域）转换为向量嵌入，并存储到向量数据库中。
      * **召回 (RAG)：** `retrieve_historical_clues` 根据 Agent 的语义查询，从向量数据库中检索最相关的历史线索，作为 **Observation** 返回给 LLM，增强其长期情境推理能力。

-----

## 🛠️ 安装与环境配置

### 1\. 软件要求

  * **操作系统：** Windows 10/11 或 Ubuntu 20.04+
  * **仿真环境：** Unreal Engine 4.27 或 5.x（取决于 AirSim 兼容性）
  * **核心库：** Python 3.8+

### 2\. AirSim 环境设置

1.  **安装 Unreal Engine：** 确保安装兼容版本的 UE。
2.  **AirSim 插件：** 从 AirSim GitHub 仓库编译并集成 AirSim 插件到您的 UE 项目中（例如：`Blocks` 场景）。
3.  **启动仿真：** 启动 Unreal Editor，打开您的项目，并点击 **Play** 开始仿真。

### 3\. Python 依赖安装

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # 或 venv\Scripts\activate

# 安装依赖库
pip install airsim numpy pandas chromadb
# 如果使用真实的 LLM API，还需安装对应的库 (e.g., openai, google-genai, anthropic)
```

-----

## 📂 代码结构

```
UAV_LLM_Agent_PoC/
├── main_agent.py          # 🚀 Agent 核心驱动：实现 '思考-行动-观察' 主循环
├── airsim_client.py       # 🔌 AirSim 基础连接和初始化
├── llm_agent_core/
│   ├── memory_manager.py  # 📚 长期记忆 (LTM)：RAG 向量数据库实现
│   └── prompt_templates.py# (省略，但应包含 CORE_PROMPT 和 FUNCTION_SCHEMAS)
├── uav_tools/             # 🛠️ 具身工具函数集
│   ├── flight_controls.py # 飞行和导航工具 (fly_to_gps, move_forward)
│   └── vision_bridge.py   # 感知工具：YOLO 模拟和像素到世界坐标转换
└── models/
    └── yolo_v8.pt         # (可选) YOLO 模型文件存放目录
```

-----

## 🚀 运行原型验证 (PoC)

1.  **确保 AirSim 运行：** 在 Unreal Editor 中启动您的 AirSim 场景。
2.  **运行主程序：** 执行 `main_agent.py` 文件。

<!-- end list -->

```bash
python main_agent.py
```

### 预期输出流程

程序将输出 Agent 的思考和行动日志：

```
============================================================
Agent 主循环启动 | 初始目标: 去坐标 (...) 附近的区域，寻找红色的标记物或失踪者。
============================================================

--- 步骤 1：规划阶段 ---
✅ Agent 思考: 任务开始，首先需要起飞到安全高度 20 米。
   -> 调用工具: takeoff ({'altitude': 20.0})
   <- 观察结果 (Observation): 起飞成功，当前高度 20.0 米。...

--- 步骤 2：规划阶段 ---
✅ Agent 思考: 已起飞，下一步是获取当前精确位置，用于规划前往目标区域的路径。
   -> 调用工具: get_current_pose ({})
   <- 观察结果 (Observation): {'latitude': 47.641..., 'longitude': -122.14...}...
   
... (流程持续，直到 Agent 调用 report_finding 或达到步数限制) ...
```

**❗ 注意：** 在 PoC 阶段，请在 `main_agent.py` 中将 `mock_llm_call` 替换为您实际的 LLM API 调用，以获得真实的智能体行为。

## 📈 后续工作和评估

项目下一步应专注于：

1.  **LLM 集成：** 将 `mock_llm_call` 替换为真实的 LLM API。
2.  **视觉模型部署：** 部署 YOLO 模型，并完善 `pixel_to_world_coordinates` 的几何计算。
3.  **性能评估：** 参照项目规划中的评估指标（搜索效率、鲁棒性、推理准确性）对 Agent 进行严格测试。
