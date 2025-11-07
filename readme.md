好的，既然 OpenFly 是底层执行模块，我们将项目命名为更具战略意义和模块化特性的名称，突出其双层智能体的架构。

我们定名为：**【AeroMind】**。

  * **Aero-**：指航空、飞行 (Aeronautics)。
  * **-Mind**：指智能、规划、思考 (Mind)。

以下是为 **AeroMind** 项目编写的全新 README 文档，突出其分层智能和对 OpenFly 的集成。

-----

# 🚀 AeroMind: 基于分层智能的无人机自主探索平台

## 目录

1.  [项目简介](https://www.google.com/search?q=%231-%E9%A1%B9%E7%9B%AE%E7%AE%80%E4%BB%8B)
2.  [架构概览：双层智能体](https://www.google.com/search?q=%232-%E6%9E%B6%E6%9E%84%E6%A6%82%E8%A7%88%E5%8F%8C%E5%B1%82%E6%99%BA%E8%83%BD%E4%BD%93)
3.  [核心功能](https://www.google.com/search?q=%233-%E6%A0%B8%E5%BF%83%E5%8A%9F%E8%83%BD)
4.  [安装与运行](https://www.google.com/search?q=%234-%E5%AE%89%E8%A3%85%E4%B8%8E%E8%BF%90%E8%A1%8C)
5.  [配置场景与记忆](https://www.google.com/search?q=%235-%E9%85%8D%E7%BD%AE%E5%9C%BA%E6%99%AF%E4%B8%8E%E8%AE%B0%E5%BF%86)
6.  [未来展望](https://www.google.com/search?q=%236-%E6%9C%AA%E6%9D%A5%E5%B1%95%E6%9C%9B)

-----

## 1\. 🌟 项目简介

**AeroMind** 是一个基于 **AirSim 仿真环境** 的高级无人机（UAV）自主探索和搜索平台。本项目核心实现了 **LLM（大语言模型）** 与 **VLA（视觉-语言-动作）模型** 的高效分层协作，解决了传统无人机在未知、复杂环境中长期规划与实时反应的矛盾。

AeroMind 借鉴了具身智能（Embodied AI）的最新研究，使无人机能够像人类一样，在宏观上进行战略思考，在微观上进行实时视觉反应。

-----

## 2\. 🧠 架构概览：双层智能体

AeroMind 采用 **Cloud/GCS-Edge Hybrid** 混合部署架构，由两个核心智能体层级构成，完美结合了长程规划和实时感知能力：

| 层级 | 名称 | 模型 | 部署位置 | 核心职责 |
| :--- | :--- | :--- | :--- | :--- |
| **高级规划层** | **LLM Planner** | GPT-4/Llama/等 (通过 OpenAI API) | 地面站 (GCS) / 云端 | **宏观决策：** 任务分解、GPS 航线计算、RAG 记忆管理、VLA 指令生成。 |
| **低级执行层** | **VLA Executor** | OpenFly VLA (模拟) | 无人机边缘 (Edge) | **微观执行：** 实时避障、精细搜索、视觉语义目标接近、低级动作控制。 |

-----

## 3\. 🛠️ 核心功能

  * **分层导航与控制：**
      * `fly_to_gps`：LLM 用于宏观、快速部署，在高空安全机动。
      * `execute_vln_instruction`：LLM 切换到 VLA 模式，执行复杂的局部视觉指令（例如：“绕过那棵树，飞到红色建筑的南侧”）。
  * **RAG 记忆与感知 (LTM)：**
      * 基于 **ChromaDB** 实现长期记忆管理。
      * `retrieve_historical_clues`：LLM 在规划前查询已探索区域和历史线索（如上次发现的物体）。
      * `update_search_map`：Agent 实时记录 VLA 发现的任何新线索或标记已探索区域。
  * **场景隔离与数据加载：**
      * 支持基于 `scene_name` 的持久化记忆库，实现不同 AirSim 场景间的独立记忆。
      * CLI 控制是否加载 **静态语义地图**（`.jsonl` 文件），用于 LLM 的先验知识。

-----

## 4\. ⚙️ 安装与运行

### 依赖项

```bash
# 必需库
pip install openai chromadb numpy airsim
# AirSim 客户端依赖：确保 AirSim/Unreal Engine 环境已配置完毕
```

### 配置 API Key

本项目依赖 OpenAI API 进行 LLM 推理。请设置您的环境变量：

```bash
export OPENAI_API_KEY="sk-..."
```

### 运行 Agent

使用命令行参数指定当前场景名称 (`--scene_name`) 和初始任务目标 (`--goal`)。

  * **默认运行 (仅使用动态记忆)：**
    LLM 将根据运行中的观测结果进行规划和记忆。

    ```bash
    python main_agent.py \
      --scene_name Neighborhood_A \
      --goal "在社区内寻找任何红色的高价值目标。" 
    ```

  * **加载静态语义地图 (开启先验知识)：**
    通过 `--load_static_map` 启用先验知识，LLM 可以利用静态地图信息进行更智能的规划。

    ```bash
    python main_agent.py \
      --scene_name Canyon_B \
      --goal "沿着峡谷底部向东飞行，寻找一座简易桥梁。" \
      --load_static_map
    ```

-----

## 5\. 📂 配置场景与记忆

### 记忆库位置

持久化记忆数据库将自动创建在：

`./chroma_dbs/<YOUR_SCENE_NAME>/`

### 静态语义地图位置

如果启用 `--load_static_map`，系统将查找：

`scene_data/seg_map/<YOUR_SCENE_NAME>_seg_map.jsonl`

请确保您的 JSONL 数据已按照规定的 **OpenFly 格式** 放置在此目录。

-----

## 6\. ✨ 未来展望

  * **真实的 VLA 模型集成：** 替换当前的 VLA 模拟器，集成量化后的 OpenFly VLA 模型，实现真正的边缘实时避障。
  * **多模态 RAG：** 允许 LLM 不仅召回文本，还能召回和分析历史图像（例如，目标图像片段）。
  * **动态环境建模：** 引入 SLAM 或 VIO，允许 Agent 在未知环境中构建和更新自身的局部地图。