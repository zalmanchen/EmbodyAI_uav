# 🚁 LLM-Agent 驱动的无人机搜索与救援仿真平台 (UAV-SAR)

## 📖 项目概览

本项目旨在构建一个高鲁棒性的具身智能体（Embodied Agent），用于在户外环境中执行复杂的搜索与救援（SAR）任务。核心优势在于采用了先进的 **双层 Agent 架构**，并直接集成了 **OpenFly-Platform** 的环境与工具链，实现了高效且高保真的视觉-语言导航（VLN）。

### 核心特性与技术栈

| 特性 | 描述 | 技术栈 |
| :--- | :--- | :--- |
| **双层 Agent 架构** | **LLM (高级规划者)** 负责长程任务分解和宏观路径规划；**OpenFly VLA 模型 (低级执行器)** 负责短程、实时、反应式的视觉-语言导航。 | LLM Function Calling, OpenFly-Agent |
| **VLN 集成基准** | 直接依赖并集成 `SHAILAB-IPEC/OpenFly-Platform` 的环境、工具链和基准数据集，确保仿真场景的高保真度和数据的语义丰富性。 | Unreal Engine **4.27** / AirSim, OpenFly 工具链 |
| **具身工具箱** | 通过 RPC 接口，将 LLM 的输出桥接到 AirSim 的物理动作、视觉感知和记忆存储。 | `airsim_client.py`, `uav_tools/` |
| **鲁棒性与稳定** | 针对 AirSim 常见的连接问题进行优化，并锁定在兼容性最佳的 UE 版本。 | 优化后的 `AirSimClient` |

-----

## 🛠️ 一、安装与环境配置

本项目依赖于 Unreal Engine 4.27、AirSim 以及 OpenFly-Platform 的工具链。此过程在 **Ubuntu 22.04** 上最为稳定。

### 1\. 软件环境要求

| 软件 | 版本要求 | 说明 |
| :--- | :--- | :--- |
| **操作系统** | Ubuntu 22.04 | 推荐用于稳定编译 UE 源码。 |
| **Unreal Engine**| **4.27** | 必须从 GitHub 源码编译。 |
| **Python** | 3.8+ | |
| **仿真插件** | AirSim Plugin | 编译集成到 OpenFly 的 UE 项目中。 |

### 2\. 获取源码

克隆本项目代码和依赖的 OpenFly-Platform 仓库：

```bash
# 1. 克隆本项目 (UAV-SAR Agent)
git clone https://your-uav-sar-repo.git uav-sar-agent
cd uav-sar-agent

# 2. 克隆 OpenFly-Platform 仓库
git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git OpenFly-Platform
```

### 3\. 环境与依赖安装 (Ubuntu 22.04)

#### A. 系统依赖

```bash
# 安装构建工具和基础依赖
sudo apt update
sudo apt install build-essential clang git cmake ninja-build
# 安装 UE 官方推荐的图形和媒体依赖
sudo apt install libsdl2-dev libxmu-dev libxi-dev libglu1-mesa-dev \
libvulkan-dev vulkan-tools libfontconfig1-dev libfreetype6-dev \
libgtk-3-dev ... # 更多依赖见完整指南
```

#### B. UE 4.27 与 AirSim 编译

请按照之前提供的 **详细安装指南** 编译 UE 4.27 源码，并运行 AirSim 的 `./setup.sh` 和 `./build.sh` 脚本。

#### C. Python 依赖

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate 

# 安装本项目核心依赖
pip install airsim numpy pandas chromadb 

# 安装 OpenFly-Platform 的依赖 (请参考 OpenFly-Platform/requirements.txt)
pip install -r OpenFly-Platform/requirements.txt
```

-----

## 🧠 二、双层 Agent 架构与集成细节

### 1\. LLM 高级规划层 (High-Level Planner)

  * **文件：** `main_agent.py`, `llm_agent_core/prompt_templates.py`
  * **职责：** 负责任务的宏观决策和调度。

| 宏观行动 | 对应的 LLM 工具 | 目标 |
| :--- | :--- | :--- |
| **长程移动** | `fly_to_gps` | 快速跨越区域，到达目标搜索范围。 |
| **信息召回** | `retrieve_historical_clues` | 利用 RAG 长期记忆辅助决策。 |
| **精细搜索** | **`execute_vln_instruction`** | 切换到低级 VLA 模式，进行视觉导航。 |

### 2\. OpenFly VLA 低级执行层 (Low-Level Executor)

这是集成的核心部分，由 `execute_vln_instruction` 工具触发。

  * **VLA 逻辑：**
    1.  LLM 调用 `execute_vln_instruction(instruction="沿着小路飞到红色房子旁边")`。
    2.  该函数加载 **OpenFly-Agent 模型**（位于 `OpenFly-Platform` 内部）。
    3.  VLA 模型实时获取 AirSim 图像和状态，进行推理。
    4.  VLA 模型输出低级控制指令（如速度向量、角速度）。
    5.  AirSim Client 执行这些指令，直到 VLA 任务子目标完成。
    6.  将 VLA 的运行结果（如 "目标已找到，坐标 X"）作为 **OBSERVATION** 返回给 LLM。

### 3\. 鲁棒性优化 (`airsim_client.py`)

集成了指数退避重试机制和多名称尝试，确保在启动时能够成功连接仿真环境并解锁无人机。

-----

## 🚀 四、运行与验证

### 1\. 启动仿真环境

1.  在 Unreal Editor (UE 4.27) 中打开 **OpenFly-Platform 提供的 AirSim 场景**。
2.  确保 AirSim 插件已启用，并按下 **Play** 按钮启动仿真。

### 2\. 运行 Agent 主程序

```bash
python main_agent.py
```

### 3\. 预期输出

观察控制台日志，确认 Agent 遵循 **“思考-行动-观察”** 循环，并正确执行了从 `fly_to_gps` 到 **`execute_vln_instruction`** 的控制权切换。

-----

## ⏭️ 五、后续工作规划

当前的 PoC 框架结构已就位，后续工作将聚焦于实现真实的 LLM/VLA 推理。

1.  **真实的 LLM API 集成：** 将 `mock_llm_call` 替换为 OpenAI/Gemini 或其他 LLM 的 Function Calling API 接口。
2.  **OpenFly-Agent 模型接入：** 详细研究 OpenFly 仓库，实现 `execute_vln_instruction` 内部的 VLA 模型加载、推理和实时 AirSim API 调用。
3.  **视觉感知模块升级：** 将 `capture_and_analyze_rgb` 升级为集成 VLM（如 LLaVA/Gemini Pro Vision）的感知模块，利用 OpenFly 的语义数据。

-----

Would you like me to provide the previously discussed detailed installation steps for UE 4.27 on Ubuntu 22.04, or is the current structure sufficient?
