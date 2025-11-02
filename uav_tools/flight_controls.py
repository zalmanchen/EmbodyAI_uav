import airsim
import time
from airsim_client import AirSimClient # 导入我们之前编写的基础客户端

# 假设全局或通过外部传入的 AirSim 客户端实例
# 在实际的 Agent 框架中，您会确保这个客户端是已连接和初始化的。
try:
    UAV_CLIENT = AirSimClient(vehicle_name="Drone1")
    UAV_CLIENT.connect_and_initialize()
except Exception as e:
    print(f"警告：AirSimClient 初始化失败，飞行控制工具将无法使用。错误: {e}")
    UAV_CLIENT = None

# --- LLM Agent 核心飞行工具 ---

def fly_to_gps(latitude: float, longitude: float, altitude_meters: float, velocity: float = 10.0) -> str:
    """
    【LLM-Agent 工具】命令无人机飞往指定的 GPS 坐标。

    参数:
        latitude (float): 目标纬度。
        longitude (float): 目标经度。
        altitude_meters (float): 目标绝对高度 (ASL)，应为正值（例如：30.0米）。
        velocity (float): 巡航速度 (米/秒)。

    返回:
        str: 飞行结果的观察报告。
    """
    if not UAV_CLIENT or not UAV_CLIENT.connected:
        return "ERROR: 飞行控制失败，AirSim 客户端未连接或未初始化。"

    # AirSim 约定: Z 轴朝下，因此高度必须取负值。
    airsim_altitude = -abs(altitude_meters) 

    try:
        current_pose = UAV_CLIENT.get_current_pose()
        
        print(f"Agent 规划飞行：从 ({current_pose['latitude']:.5f}, {current_pose['longitude']:.5f})")
        print(f"            飞往：({latitude:.5f}, {longitude:.5f})，高度 {altitude_meters:.1f}m，速度 {velocity:.1f}m/s")
        
        UAV_CLIENT.client.moveToGPSAsync(
            latitude, 
            longitude, 
            airsim_altitude, 
            velocity,
            timeout_sec=60
        ).join()
        
        # 飞行完成后，获取最终位置进行确认
        final_pose = UAV_CLIENT.get_current_pose()
        
        # 简单的检查以验证飞行是否接近目标
        lat_diff = abs(final_pose['latitude'] - latitude)
        lon_diff = abs(final_pose['longitude'] - longitude)
        
        if lat_diff < 0.00005 and lon_diff < 0.00005: # 这是一个非常简化的成功判定
            return f"OBSERVATION: 成功飞抵目标坐标 ({latitude:.5f}, {longitude:.5f})，当前高度 {final_pose['altitude_meters']:.1f} 米。"
        else:
            return f"WARNING: 飞行完成，但当前位置与目标有偏差。当前坐标: ({final_pose['latitude']:.5f}, {final_pose['longitude']:.5f})"

    except Exception as e:
        # 如果超时或其他错误，解除锁定并报告
        UAV_CLIENT.client.armDisarm(False)
        return f"ERROR: 飞往 GPS 目标失败：{e}"


def move_forward(distance: float = 5.0, velocity: float = 5.0) -> str:
    """
    【LLM-Agent 工具】向前飞行指定的距离 (米)。
    用于近距离检查和简单避障。
    """
    if not UAV_CLIENT or not UAV_CLIENT.connected:
        return "ERROR: 飞行控制失败，AirSim 客户端未连接或未初始化。"
    
    try:
        # 使用 moveByVelocityAsync 实现相对移动，但在 AirSim 中，
        # moveByVelocityAsync 默认是在世界坐标系中，需要结合姿态进行转换。
        # 简化做法：使用 moveByDistanceAsync（相对当前朝向）
        UAV_CLIENT.client.moveByDistanceAsync(distance, velocity).join()
        
        return f"OBSERVATION: 成功向前移动 {distance:.1f} 米。"
    except Exception as e:
        return f"ERROR: 向前移动失败：{e}"


def set_yaw(yaw_degrees: float, velocity: float = 5.0) -> str:
    """
    【LLM-Agent 工具】将无人机朝向旋转到指定的绝对偏航角（Yaw）。
    用于全景扫描或对齐目标。
    """
    if not UAV_CLIENT or not UAV_CLIENT.connected:
        return "ERROR: 飞行控制失败，AirSim 客户端未连接或未初始化。"
    
    try:
        # AirSim 中的 Yaw 是以弧度计算的，我们需要提供度数并转换为弧度。
        yaw_rad = np.radians(yaw_degrees)
        
        # is_degrees=False 表示我们提供的是弧度
        UAV_CLIENT.client.rotateToYawAsync(yaw_rad, velocity, is_degrees=False).join()
        
        return f"OBSERVATION: 成功将无人机朝向旋转至 {yaw_degrees:.1f} 度 (绝对偏航角)。"
    except Exception as e:
        return f"ERROR: 旋转朝向失败：{e}"


# --- 示例用法 (用于测试该文件功能) ---

if __name__ == "__main__":
    if UAV_CLIENT and UAV_CLIENT.connected:
        
        # 1. 确保已起飞
        UAV_CLIENT.takeoff(altitude=10.0)
        time.sleep(2)
        
        # 2. 测试 GPS 飞行 (请替换为您的 AirSim 地图中的有效 GPS 坐标)
        TEST_LAT = 47.641467 # 示例坐标
        TEST_LON = -122.140135 # 示例坐标
        
        print("\n--- 测试 fly_to_gps ---")
        fly_result = fly_to_gps(TEST_LAT, TEST_LON, altitude_meters=15.0, velocity=15.0)
        print(f"飞行测试结果: {fly_result}")
        time.sleep(2)
        
        # 3. 测试 旋转朝向
        print("\n--- 测试 set_yaw ---")
        yaw_result = set_yaw(90.0)
        print(f"旋转测试结果: {yaw_result}")
        time.sleep(2)
        
        # 4. 测试 相对移动
        print("\n--- 测试 move_forward ---")
        move_result = move_forward(distance=10.0)
        print(f"移动测试结果: {move_result}")
        
        # 5. 降落清理
        UAV_CLIENT.land()
    else:
        print("无法运行飞行控制测试，请检查 AirSim 客户端初始化是否成功。")