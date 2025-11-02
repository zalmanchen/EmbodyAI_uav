import airsim
import time
import numpy as np

# --- 核心 AirSim 客户端类 ---

class AirSimClient:
    """
    封装 AirSim 客户端的类，提供连接、初始化和基本飞行控制功能。
    """
    
    def __init__(self, vehicle_name="Drone1"):
        """初始化 AirSim 客户端并尝试连接。"""
        self.vehicle_name = vehicle_name
        self.client = airsim.MultirotorClient()
        self.connected = False

    def connect_and_initialize(self):
        """
        连接到 AirSim 仿真环境，并准备无人机进行飞行。
        """
        try:
            print(f"尝试连接到 AirSim 客户端...")
            self.client.confirmConnection()
            self.connected = True
            print("连接成功！")

            # 重置环境状态
            self.client.reset()
            time.sleep(0.5)
            
            # 启用 API 控制
            self.client.enableApiControl(True, self.vehicle_name)
            
            # 解锁无人机（上电，准备飞行）
            self.client.armDisarm(True, self.vehicle_name)
            print(f"无人机 '{self.vehicle_name}' 已解锁并准备就绪。")
            
            return True
        except Exception as e:
            print(f"连接 AirSim 或初始化失败: {e}")
            self.connected = False
            return False

    def takeoff(self, altitude=20.0):
        """
        命令无人机垂直起飞并悬停在指定高度。
        """
        if not self.connected:
            return "错误：未连接到 AirSim。"
        
        try:
            print(f"无人机起飞至 {altitude} 米...")
            # AirSim 的 Z 坐标是朝下的，所以高度为负值
            self.client.takeoffAsync(timeout_sec=5).join() 
            
            # 确保到达目标高度（AirSim 默认起飞到一个安全高度，这里明确飞到指定高度）
            self.client.moveToZAsync(-altitude, 5).join()
            print(f"无人机已到达 {altitude} 米并悬停。")
            return f"起飞成功，当前高度 {altitude} 米。"
        except Exception as e:
            print(f"起飞失败: {e}")
            self.client.armDisarm(False, self.vehicle_name)
            return f"错误：起飞失败 - {e}"

    def land(self):
        """
        命令无人机在当前位置降落并锁定。
        """
        if not self.connected:
            return "错误：未连接到 AirSim。"
        
        try:
            print("无人机开始降落...")
            self.client.landAsync().join()
            
            # 降落后，解除锁定（断电）
            self.client.armDisarm(False, self.vehicle_name)
            print("无人机已安全降落并锁定。")
            return "降落成功。"
        except Exception as e:
            print(f"降落失败: {e}")
            return f"错误：降落失败 - {e}"
            
    def get_current_pose(self):
        """
        获取无人机的当前 GPS 坐标和姿态。
        """
        if not self.connected:
            return None

        state = self.client.getMultirotorState()
        gps = state.gps_location
        
        # 将四元数转换为欧拉角，获取 Yaw (朝向)
        orientation = state.kinematics_estimated.orientation
        pitch, roll, yaw = airsim.to_euler_angles(orientation)
        
        # 封装为 Agent 可读的格式
        pose = {
            "latitude": gps.latitude,
            "longitude": gps.longitude,
            "altitude_meters": -gps.altitude, # AirSim Z朝下，我们将其转换为正的绝对高度
            "yaw_degrees": np.degrees(yaw)
        }
        
        return pose

# --- 示例用法 (用于测试该文件功能) ---

if __name__ == "__main__":
    # 确保 AirSim 仿真环境在运行，并加载了您的场景
    
    uav_client = AirSimClient(vehicle_name="Drone1")
    
    if uav_client.connect_and_initialize():
        # 测试 1: 获取初始姿态
        initial_pose = uav_client.get_current_pose()
        print(f"\n初始姿态: {initial_pose}")

        # 测试 2: 起飞
        takeoff_result = uav_client.takeoff(altitude=15.0)
        print(f"起飞结果: {takeoff_result}")
        time.sleep(2)

        # 测试 3: 获取起飞后的姿态
        current_pose = uav_client.get_current_pose()
        print(f"当前姿态: {current_pose}")

        # 测试 4: 降落
        land_result = uav_client.land()
        print(f"降落结果: {land_result}")
    else:
        print("\n请检查 AirSim 仿真环境是否已启动。")