# 📁 airsim_client.py

import airsim
import time
from typing import Dict, Any

class AirSimClient:
    """
    AirSim RPC 客户端封装，处理连接、初始化和基础控制。
    """
    
    def __init__(self, vehicle_name: str = "Drone1"):
        self.vehicle_name = vehicle_name
        self.client = None
        self.max_retries = 5

    def connect_and_initialize(self) -> bool:
        """
        连接 AirSim 仿真器，并进行初始化设置。
        """
        print("尝试连接 AirSim RPC 服务器...")
        
        for attempt in range(self.max_retries):
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                print(f"✅ AirSim 连接成功 (尝试 {attempt + 1}/{self.max_retries})。")
                
                # 尝试重置环境并启用 API 控制
                self._reset_and_enable_api_control()
                return True
                
            except Exception as e:
                print(f"连接失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2 ** attempt)  # 指数退避等待
                
        print("❌ FATAL ERROR: 无法连接 AirSim 服务器，请检查仿真环境是否运行。")
        return False

    def _reset_and_enable_api_control(self):
        """重置环境，并尝试启用 API 控制和解锁。"""
        
        # 1. 重置环境 (重要：必须在飞行前重置)
        self.client.reset() 
        time.sleep(0.5) # 等待环境稳定
        
        success = False
        
        # 2. 尝试使用默认车辆名或空字符串启用 API 控制
        vehicle_names_to_try = [self.vehicle_name, ""]
        
        for name in vehicle_names_to_try:
            try:
                self.client.enableApiControl(True, name)
                self.client.armDisarm(True, name)
                
                # 验证是否成功
                if self.client.isApiControlEnabled(name) and self.client.getMultirotorState(name).landed_state == airsim.LandedState.Armed:
                    self.vehicle_name = name # 记录成功的车辆名
                    print(f"✅ API 控制和解锁成功，使用的车辆名为: '{name}'")
                    success = True
                    break
            except Exception as e:
                # print(f"尝试车辆名 '{name}' 失败: {e}")
                pass # 静默失败，继续尝试下一个名称
        
        if not success:
            raise Exception("无法启用 API 控制或解锁无人机。")


    # --- 基础控制 API ---

    def takeoff(self, altitude: float) -> str:
        """执行起飞到指定高度。"""
        print(f"执行起飞到 {altitude}m...")
        self.client.takeoffAsync(timeout_sec=5).join()
        
        # 飞到指定高度，以确保高度精确
        z = self.client.getMultirotorState(self.vehicle_name).position.z_val
        if z > -altitude + 1: # AirSim NED 坐标系下，z为负值表示高度
            self.client.moveToZAsync(-altitude, 2, vehicle_name=self.vehicle_name).join()
            
        return f"OBSERVATION: 无人机起飞成功，位于高度 {altitude:.2f} 米。"

    def land(self) -> str:
        """执行降落。"""
        print("执行降落...")
        self.client.landAsync(timeout_sec=5).join()
        self.client.armDisarm(False, self.vehicle_name)
        return "OBSERVATION: 无人机已安全降落并解除锁定。"

    def get_current_pose(self) -> str:
        """获取并返回无人机当前的 GPS 坐标和姿态（NED 坐标系）。"""
        state = self.client.getMultirotorState(self.vehicle_name)
        gps = self.client.getGpsLocation(self.vehicle_name)
        
        # AirSim NED 坐标系下，Z 为负值，需要转换为正高度
        altitude_meters = -state.position.z_val 
        
        # 姿态（四元数）
        orientation = state.kinematics_estimated.orientation
        
        pose_data = {
            "latitude": gps.latitude,
            "longitude": gps.longitude,
            "altitude_meters": altitude_meters,
            "orientation_w_x_y_z": [orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]
        }
        
        return f"OBSERVATION: 当前姿态：Lat={gps.latitude:.6f}, Lon={gps.longitude:.6f}, Alt={altitude_meters:.2f}m. 原始数据: {pose_data}"

if __name__ == "__main__":
    # 验证客户端
    client_test = AirSimClient()
    if client_test.connect_and_initialize():
        print(client_test.takeoff(20))
        time.sleep(3)
        print(client_test.get_current_pose())
        time.sleep(3)
        print(client_test.land())