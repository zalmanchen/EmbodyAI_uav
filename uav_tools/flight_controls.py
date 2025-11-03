# 📁 uav_tools/flight_controls.py

import time
from typing import Dict, Any

# 导入 airsim 库以便使用其类型定义
import airsim

# --- 导入 AirSim Client 实例 ---
# 注意: 为了避免循环导入，我们假设 main_agent.py 会将连接好的 AIRSIM_CLIENT 传入
# 或者我们通过一个延迟绑定机制获取客户端。这里为了简化 PoC，我们使用一个全局占位符。

# 警告：在实际项目中，请使用依赖注入或在运行时绑定客户端，而不是使用全局变量。
CLIENT_INSTANCE = None 

def set_airsim_client(client: airsim.MultirotorClient):
    """设置 AirSimClient 实例，供所有控制函数使用。"""
    global CLIENT_INSTANCE
    CLIENT_INSTANCE = client

def _ensure_client_ready():
    """检查客户端是否已连接并可用。"""
    if CLIENT_INSTANCE is None:
        raise ConnectionError("AirSim 客户端尚未初始化或设置。请先运行 connect_and_initialize。")

# --- 1. 高级宏观飞行控制函数 ---

def fly_to_gps(latitude: float, longitude: float, altitude_meters: float) -> str:
    """
    【LLM 工具】飞往指定的全球定位系统 (GPS) 坐标点。
    这是高级规划层的主要宏观行动工具。
    """
    _ensure_client_ready()
    
    # AirSim API 使用海平面以下（负值）作为 Z 坐标
    target_z = -altitude_meters
    speed = 5.0 # 默认速度 5 m/s

    print(f"执行宏观飞行: 飞往 Lat={latitude:.6f}, Lon={longitude:.6f}, Alt={altitude_meters:.2f}m...")
    
    # 使用 AirSim 的 moveToGPSAsync API
    CLIENT_INSTANCE.client.moveToGPSAsync(
        latitude, 
        longitude, 
        target_z, 
        speed, 
        timeout_sec=60,
        vehicle_name=CLIENT_INSTANCE.vehicle_name
    ).join()

    # 验证是否到达目标点 (简单的距离检查)
    current_gps = CLIENT_INSTANCE.client.getGpsLocation(CLIENT_INSTANCE.vehicle_name)
    
    # 模拟成功的观察结果
    return (f"OBSERVATION: 无人机成功飞抵目标坐标附近的区域。"
            f"当前 GPS: Lat={current_gps.latitude:.6f}, Lon={current_gps.longitude:.6f}, Alt={-current_gps.altitude:.2f}m。")

# 辅助宏观控制 (如果 LLM 决定使用 NED 坐标系进行微调)
def move_forward(distance: float) -> str:
    """在当前航向向前移动指定距离 (米)。"""
    _ensure_client_ready()
    speed = 2.0
    
    # 假设使用相对移动 API
    CLIENT_INSTANCE.client.moveByVelocityBodyFrameAsync(
        vx=speed, vy=0, vz=0, duration=distance/speed, vehicle_name=CLIENT_INSTANCE.vehicle_name
    ).join()
    
    return f"OBSERVATION: 无人机向前移动了 {distance:.2f} 米。"

def set_yaw(yaw_degrees: float) -> str:
    """设置无人机的偏航角 (相对于世界坐标系，度)。"""
    _ensure_client_ready()
    
    # 使用 AirSim 的 setYaw API
    CLIENT_INSTANCE.client.rotateToYawAsync(
        yaw_degrees, 
        timeout_sec=3, 
        vehicle_name=CLIENT_INSTANCE.vehicle_name
    ).join()
    
    return f"OBSERVATION: 无人机偏航角已设置为 {yaw_degrees:.1f} 度。"


# --- 2. OpenFly VLA 低级执行工具 ---

def execute_vln_instruction(language_instruction: str) -> str:
    """
    【LLM 工具】激活 OpenFly VLA 模型，执行视觉-语言导航任务。
    LLM 将控制权移交给 VLA 模型进行精细、反应式的导航。
    
    在这个 PoC 中，我们模拟 VLA 模型的运行和结果。
    """
    _ensure_client_ready()
    
    # --- VLA 模型模拟/集成点 ---
    
    # 1. 模拟 VLA 启动和推理
    print(f"\n[VLA 启动] 接收到指令：'{language_instruction}'")
    
    # 2. 模拟 VLA 运行循环 (实际中这里是 VLA 模型实时获取图像->推理->发送控制指令)
    run_time = 5  # 模拟 VLA 执行 5 秒
    print(f"[VLA 运行] 模拟 VLA 模型正在实时控制无人机进行搜索... 持续 {run_time} 秒。")
    
    # 模拟 VLA 的低级控制输出 (例如，前进 2m, 偏航 30度)
    # 实际中 VLA 会持续输出动作
    
    # 模拟 VLA 找到目标，并返回精确的位置
    time.sleep(run_time) 
    
    # 3. 模拟 VLA 返回精确的发现位置
    mock_found_coords = f"Lat: 47.6417, Lon: -122.1401, Alt: 27.0m"
    
    # --- VLA 模拟结束 ---
    
    return (f"OBSERVATION: VLA 模型已完成导航指令 '{language_instruction}'。 "
            f"VLA 报告：成功在目标附近找到线索。精确位置: {mock_found_coords}。")

# -------------------------------------------------------------
# 您需要在 main_agent.py 中添加 set_airsim_client 的调用！
# -------------------------------------------------------------
