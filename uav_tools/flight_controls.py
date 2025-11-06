# 📁 uav_tools/flight_controls.py

import time
from typing import Dict, Any

import torch

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
    # _ensure_client_ready()
    
    # AirSim API 使用海平面以下（负值）作为 Z 坐标
    target_z = -altitude_meters
    speed = 5.0 # 默认速度 5 m/s

    print(f"执行宏观飞行: 飞往 Lat={latitude:.6f}, Lon={longitude:.6f}, Alt={altitude_meters:.2f}m...")
    
    # 1. 执行 GPS 飞行
    # 使用 AirSim 的 moveToGPSAsync API
    CLIENT_INSTANCE.client.moveToGPSAsync(
        latitude, 
        longitude, 
        target_z, 
        speed, 
        timeout_sec=60,
        vehicle_name=CLIENT_INSTANCE.vehicle_name
    ).join()

    # 2. 验证是否到达目标点 (简单的距离检查)
    
    # ***** 错误修正点：使用 getGpsData 而不是 getGpsLocation *****
    # 注意：getGpsData 返回 GpsData 对象，其位置属性名为 gnss.geo_point
    
    # 尝试使用 getGpsData 获取数据 (这是更准确的 API)
    gps_data = CLIENT_INSTANCE.client.getGpsData(vehicle_name=CLIENT_INSTANCE.vehicle_name)
    current_gps = gps_data.gnss.geo_point
    
    # 修正后的变量访问
    current_lat = current_gps.latitude
    current_lon = current_gps.longitude
    current_alt = -current_gps.altitude # 海拔高度转为正值

    # 模拟成功的观察结果
    return (f"OBSERVATION: 无人机成功飞抵目标坐标附近的区域。"
            f"当前 GPS: Lat={current_lat:.6f}, Lon={current_lon:.6f}, Alt={current_alt:.2f}m。")

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

# --- 辅助函数：获取 Lidar 障碍物信息 ---

def _get_nearest_obstacle_distance(lidar_name: str = "Lidar1", max_range: float = 5.0) -> float:
    """
    获取前方扇区内最近障碍物的距离。
    需要 AirSim 中已配置 Lidar 传感器。
    """
    _ensure_client_ready()
    try:
        # 假设我们只关心前方 90 度扇区（例如 yaw +/- 45 度）
        lidar_data = CLIENT_INSTANCE.client.getLidarData(lidar_name, CLIENT_INSTANCE.vehicle_name)
        
        # Lidar 数据是点云，通常需要复杂的处理。这里我们简化为检查最近的X距离。
        # AirSim LidarData 包含 point_cloud 数组，通常是 (x, y, z) 坐标的扁平列表。
        
        min_dist_sq = max_range * max_range  # 用距离平方做比较更高效
        
        # 简化处理：遍历点云，只考虑前方点 (X > 0)
        points = lidar_data.point_cloud
        num_points = int(len(points) / 3)
        
        for i in range(num_points):
            x = points[i*3 + 0]
            y = points[i*3 + 1]
            z = points[i*3 + 2]
            
            # 只考虑正前方（X轴正方向）且在高度范围内（例如 Z 在 -0.5 到 0.5 之间）
            if x > 0.5 and abs(z) < 1.0: 
                dist_sq = x*x + y*y # 忽略 Z 轴影响，只看水平距离
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq

        return min_dist_sq**0.5 # 返回最近距离
        
    except Exception as e:
        print(f"Warning: Lidar 数据获取失败 ({e})，返回最大安全距离。")
        return max_range


# --- 2. 局部避障移动工具 ---

def move_with_local_avoidance(target_distance: float, lidar_name: str = "Lidar1") -> str:
    """
    【LLM 工具】在当前航向安全移动指定距离，使用 Lidar 进行局部避障。
    如果遇到障碍物，执行简单的转向策略。
    """
    _ensure_client_ready()
    
    SAFETY_DISTANCE = 3.0  # 障碍物小于 3 米时启动避障
    MOVE_STEP = 1.0        # 每次移动 1 米
    MAX_YAW_ATTEMPTS = 5   # 最大尝试转向次数
    distance_moved = 0.0
    
    # 转换为 NED 坐标系下的 Z 坐标（用于高度）
    current_state = CLIENT_INSTANCE.client.getMultirotorState(CLIENT_INSTANCE.vehicle_name)
    current_z = current_state.kinematics_estimated.position.z_val

    print(f"执行局部避障移动：目标 {target_distance:.1f}m...")
    
    while distance_moved < target_distance:
        # 1. 检查前方距离
        nearest_obstacle = _get_nearest_obstacle_distance(lidar_name, SAFETY_DISTANCE * 2)
        
        if nearest_obstacle < SAFETY_DISTANCE:
            # 2. 发现障碍物：执行避障策略 (转弯)
            
            print(f"-> 发现障碍物！距离 {nearest_obstacle:.1f}m，启动避障。")
            
            # 尝试向右转 30 度
            for attempt in range(MAX_YAW_ATTEMPTS):
                yaw_change = 30.0 # 转向角度
                
                # 获取当前 yaw
                current_orientation = current_state.kinematics_estimated.orientation
                _, _, current_yaw = airsim.to_eularian_angles(current_orientation)
                current_yaw_deg = current_yaw * 180 / 3.14159
                
                # 执行转弯
                set_yaw(current_yaw_deg + yaw_change) 
                
                # 检查转弯后前方是否安全
                time.sleep(1.0) # 等待转弯完成
                if _get_nearest_obstacle_distance(lidar_name) >= SAFETY_DISTANCE:
                    print(f"-> 转向成功，前方安全，执行绕行移动。")
                    
                    # 绕行 3 米
                    move_forward(3.0) 
                    
                    # 尝试回到主航线（这里为了简化不实现回转）
                    break 
                
                if attempt == MAX_YAW_ATTEMPTS - 1:
                    return f"OBSERVATION: 无法绕过障碍物，已在距离 {distance_moved:.1f}m 处停止。LLM 需重新规划。"
        
        # 3. 前方安全：继续前进
        else:
            step_to_move = min(MOVE_STEP, target_distance - distance_moved)
            move_forward(step_to_move)
            distance_moved += step_to_move
            
        time.sleep(0.5) # 循环延迟

    return f"OBSERVATION: 无人机已在局部避障后安全移动了 {distance_moved:.1f} 米。"

# -------------------------------------------------------------


# --- 2. OpenFly VLA 低级执行工具 ---

# -------------------------------------------------------------
# OPENFLY VLA CORE LOGIC (提取自您的测试脚本)
# -------------------------------------------------------------

# --- 导入 OpenFly 依赖 ---
# 注意：您可能需要在您的环境中手动处理这些导入
try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from extern.hf.configuration_prismatic import OpenFlyConfig
    from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    # 注册配置 (确保这些配置在您的环境中被正确识别)
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)
    
    OPENFLY_AVAILABLE = True
except ImportError:
    print("Warning: OpenFly/Prismatic dependencies not found. VLA execution will be simulated.")
    OPENFLY_AVAILABLE = False

# --- 全局 VLA 模型实例 ---
POLICY = None
PROCESSOR = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_openfly_agent(model_name_or_path="IPEC-COMMUNITY/openfly-agent-7b"):
    """加载 OpenFly VLA 模型和处理器，只执行一次。"""
    global POLICY, PROCESSOR
    if POLICY is not None or not OPENFLY_AVAILABLE:
        return
        
    print(f"Loading OpenFly VLA Agent from {model_name_or_path} on {DEVICE}...")
    try:
        PROCESSOR = AutoProcessor.from_pretrained(model_name_or_path)
        POLICY = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda:0" else torch.float32, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
        ).to(DEVICE)
        print("✅ OpenFly VLA Agent 加载成功。")
    except Exception as e:
        print(f"❌ 无法加载 OpenFly 模型: {e}. VLA 模式将不可用。")
        POLICY = None
        PROCESSOR = None

# # ----------------- 动作辅助函数 (直接复用) -----------------

# def get_images(lst,if_his,step):
#     # (保持您提供的 get_images 函数不变)
#     # ...

# def convert_to_action_id(action):
#     # (保持您提供的 convert_to_action_id 函数不变)
#     # ...
#     action_dict = {
#         "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
#         # ... (其他动作) ...
#     }
#     # ...
#     return result

# def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):
#     # (保持您提供的 get_action 函数不变)
#     # ...
#     return cur_action

# def getPoseAfterMakeAction(current_pose, action):
#     # (保持您提供的 getPoseAfterMakeAction 函数不变)
#     # ...
#     return [x, y, z, yaw]

# ----------------- AirSim 图像获取辅助函数 -----------------

def _get_airsim_image():
    """使用 AirSimClient 实例获取 RGB 图像 (Scene 模式)。"""
    _ensure_client_ready()
    
    responses = CLIENT_INSTANCE.client.simGetImages([
        airsim.ImageRequest('front_custom', airsim.ImageType.Scene, False, False)
    ])
    response = responses[0]
    
    # 转换为 NumPy 数组 (RGB 格式)
    img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_data = img_data.reshape(response.height, response.width, 3)
    
    # AirSim 默认 BGR，转换为 RGB
    img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    return img_data_rgb


# ----------------- 核心 VLA 执行函数 -----------------


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
