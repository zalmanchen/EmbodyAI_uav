import airsim
import numpy as np
import json
from PIL import Image
from .airsim_client import AirSimClient # 假设已导入基础客户端

# 假设全局或通过外部传入的 AirSim 客户端实例
try:
    UAV_CLIENT = AirSimClient(vehicle_name="Drone1")
    # 确保客户端已连接，但不在 vision_bridge 中重复初始化
except Exception:
    UAV_CLIENT = None

# --- 辅助函数：坐标转换（简化版） ---

def pixel_to_world_coordinates(pixel_x: int, pixel_y: int, depth_data: np.ndarray) -> dict:
    """
    【核心几何函数】根据像素坐标和深度数据，计算目标在 AirSim 世界中的 GPS 坐标。
    
    注意：这是一个高度简化的 PoC 实现。实际实现需要复杂的相机内参、
    外参（姿态）和四元数到旋转矩阵的转换。
    """
    if not UAV_CLIENT or not UAV_CLIENT.connected:
        return {"error": "未连接客户端"}

    try:
        # 1. 从深度图中获取距离 (Z)
        # 深度图数据需要正确的解析，这里直接读取像素值
        depth_val = depth_data[pixel_y, pixel_x]
        
        # 2. 获取无人机当前姿态（用于转换）
        drone_pose = UAV_CLIENT.client.simGetVehiclePose()
        drone_gps = UAV_CLIENT.client.getGpsLocation()
        
        # 3. 模拟距离的简单 GPS 偏移计算 (仅为 PoC)
        # 假设目标距离无人机 10 米，并位于图像中心
        if depth_val < 1.0 or depth_val > 500:
             # 如果深度值异常，假设一个近似值
            distance_m = 10.0
        else:
            distance_m = depth_val
        
        # 简单的线性偏移估算（高度依赖于深度值和相机俯仰角）
        # 假设 10米移动大约对应 0.00009度的经纬度变化
        offset_lat = 0.00009 * (distance_m / 10.0) 
        offset_lon = 0.0 # 假设目标在正前方
        
        # 实际应根据无人机 Yaw 角和 Pitch 角计算更复杂的偏移
        
        # 模拟计算最终目标 GPS 坐标
        target_gps = {
            "latitude": drone_gps.latitude + offset_lat,
            "longitude": drone_gps.longitude + offset_lon,
            # 高度估算：用无人机当前高度减去目标深度（再次简化）
            "altitude_meters": -drone_gps.altitude - distance_m
        }
        
        return target_gps
        
    except Exception as e:
        return {"error": f"坐标转换函数错误: {e}"}

# --- LLM Agent 核心感知工具 ---

def capture_and_analyze_rgb(target_description: str, camera_name: str = "front_center") -> str:
    """
    【LLM-Agent 工具】拍摄 RGB 和深度图像，运行 YOLO 模型，并返回带世界坐标的结果。
    
    参数:
        target_description (str): Agent 正在寻找的目标的描述 (用于指导 YOLO/VLM)。
        camera_name (str): AirSim 中相机的名称。
        
    返回:
        str: 包含检测结果的 JSON 字符串 (作为 Agent 的 Observation)。
    """
    if not UAV_CLIENT or not UAV_CLIENT.connected:
        return json.dumps({"status": "ERROR", "details": "未连接客户端，无法捕获图像"})

    try:
        # 1. 从 AirSim 获取 RGB 和 深度图像
        responses = UAV_CLIENT.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),  # RGB (Scene)
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False) # 深度
        ])
        
        # 2. 解析图像数据
        # 2a. RGB 图像（用于 YOLO 输入）
        rgb_response = responses[0]
        # rgb_img = Image.frombuffer("RGB", (rgb_response.width, rgb_response.height), rgb_response.image_data_uint8, "raw", "RGB", 0, 1)
        
        # 2b. 深度数据（用于坐标转换）
        depth_response = responses[1]
        depth_data = np.array(depth_response.image_data_float, dtype=np.float32).reshape(depth_response.height, depth_response.width)

        
        # 3. 运行 YOLO 推理（此处为 PoC 模拟）
        # 实际代码: yolo_results = run_yolo_inference(rgb_img, target_description)
        # 
        # 模拟结果：检测到一个符合 target_description 的物体，位于图像中心
        simulated_detections = []
        if "标记物" in target_description or "背包" in target_description:
             simulated_detections.append({
                "class_name": "target_found",
                "pixel_x": rgb_response.width // 2, # 图像中心
                "pixel_y": rgb_response.height // 2, 
                "confidence": 0.98
            })

        detected_objects_summary = []
        
        for result in simulated_detections:
            # 4. 执行坐标转换
            world_coords = pixel_to_world_coordinates(
                result["pixel_x"], 
                result["pixel_y"], 
                depth_data
            )
            
            detected_objects_summary.append({
                "object_type": result["class_name"],
                "confidence": result["confidence"],
                "gps_location": world_coords
            })

        if not detected_objects_summary:
            return json.dumps({"status": "SUCCESS", "details": f"未检测到符合 '{target_description}' 的物体"})
        else:
            return json.dumps({"status": "SUCCESS", "details": detected_objects_summary})
            
    except Exception as e:
        return json.dumps({"status": "ERROR", "details": f"感知工具运行失败: {e}"})


# --- 示例用法 (用于测试该文件功能) ---

if __name__ == "__main__":
    if UAV_CLIENT and UAV_CLIENT.connect_and_initialize():
        print("\n--- 测试 capture_and_analyze_rgb 工具 ---")
        
        # 模拟 Agent 调用，寻找“红色背包”
        result_json = capture_and_analyze_rgb("寻找红色背包")
        
        print("\nAgent 观察结果 (JSON):")
        print(result_json)
        
        # 检查解析结果
        result = json.loads(result_json)
        if result['status'] == 'SUCCESS' and result['details']:
            print("\n✅ 感知成功：已生成带世界坐标的结构化信息。")
            if isinstance(result['details'], list):
                print(f"检测到物体类型: {result['details'][0]['object_type']}")
                print(f"估算 GPS 坐标: {result['details'][0]['gps_location']}")
        else:
            print(f"\n❌ 感知失败或未检测到目标。")
            
        UAV_CLIENT.land()
    else:
        print("\n无法运行感知工具测试，请确保 AirSim 正在运行。")