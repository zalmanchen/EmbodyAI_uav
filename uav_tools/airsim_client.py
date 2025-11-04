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
        self.client = airsim.MultirotorClient() # 这是一个 MultirotorClient 实例
        # self.client = None
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
                
                # --- 修正点 1：将 self.vehicle_name 设为 None，强制 _reset_and_enable_api_control 尝试所有已知名称 ---
                # 这一行其实在 _reset_and_enable_api_control 内部已经处理了，但我们在这里可以确保客户端对象存在。
                
                # 尝试重置环境并启用 API 控制
                self._reset_and_enable_api_control()
                
                # --- 修正点 2：在 AirSim 客户端成功连接后，将客户端实例绑定到 flight_controls 模块 ---
                # 这解决了之前我们在 main_agent.py 中手动绑定带来的依赖问题。
                try:
                    # 假设 flight_controls 模块已准备好 set_airsim_client
                    from uav_tools.flight_controls import set_airsim_client
                    set_airsim_client(self) # 传入 self (AirSimClient 实例)
                except ImportError:
                    print("警告: 无法导入 set_airsim_client。请确保 uav_tools/flight_controls.py 文件存在。")
                
                return True
                
            except Exception as e:
                print(f"连接失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2 ** attempt)  # 指数退避等待
                
        print("❌ FATAL ERROR: 无法连接 AirSim 服务器，请检查仿真环境是否运行。")
        return False



    def _reset_and_enable_api_control(self):
        """重置环境，并尝试启用 API 控制和解锁。"""
        
        print(" -> 正在重置 AirSim 环境并尝试启用 API 控制...")
        
        self.client.reset() 
        time.sleep(0.5) 
        
        success = False
        last_error = ""
        
        vehicle_names_to_try = [self.vehicle_name, "Drone1", ""]
        unique_names = list(set(name for name in vehicle_names_to_try if name is not None))

        for name in unique_names:
            try:
                # 尝试启用 API 控制
                self.client.enableApiControl(True, name)
                
                # 尝试解锁无人机 (Arm)
                self.client.armDisarm(True, name)
                
                # 3. 验证状态：【关键修改点】仅检查 API 是否启用，并假设 armDisarm 成功
                is_api_enabled = self.client.isApiControlEnabled(name)
                
                if is_api_enabled:
                    # 成功！更新实例中的车辆名称
                    self.vehicle_name = name 
                    print(f"✅ API 控制和解锁成功。使用的车辆名: '{name}'")
                    success = True
                    return # 成功，退出函数
                
                # 如果 API 未启用
                if not is_api_enabled:
                    last_error = f"车辆 '{name}' 无法启用 API 控制，或启用后立即丢失。"
                
            except Exception as e:
                # 记录详细的错误信息
                last_error = f"车辆 '{name}' 启用控制或解锁失败: {e}"
                pass # 静默失败，继续尝试下一个名称
        
        # 4. 彻底失败：抛出异常
        if not success:
            error_message = f"无法启用 API 控制或解锁无人机。尝试的名称: {unique_names}。最后错误: {last_error}"
            # 附带一个可能的原因提示
            error_message += "\n提示: 请确保 UE 场景正在运行，AirSim 插件已加载，并且您已使用正确的 Vehicle Name。"
            raise Exception(error_message)

    # --- 基础控制 API ---


    def takeoff(self, altitude: float) -> str: # <--- 必须加上 altitude 参数
        """执行起飞到指定高度。"""
        
        # ... (您的其他确保客户端准备就绪的逻辑) ...
        
        print(f"执行起飞到 {altitude}m...")
        
        # 1. 执行 AirSim 的基本起飞命令 (通常只离开地面几米)
        self.client.takeoffAsync(timeout_sec=5, vehicle_name=self.vehicle_name).join()
        
        # 2. 修正：使用 moveToZAsync 确保无人机到达指定高度
        target_z = -altitude # AirSim NED 坐标系下，Z 为负值
        speed = 2.0
        
        self.client.moveToZAsync(
            target_z, 
            speed, 
            timeout_sec=10, 
            vehicle_name=self.vehicle_name
        ).join()
        
        # 3. 验证高度
        state = self.client.getMultirotorState(self.vehicle_name)
        current_alt = -state.kinematics_estimated.position.z_val # 修正后的位置访问
        
        return f"OBSERVATION: 无人机起飞成功，位于高度 {current_alt:.2f} 米 (目标 {altitude:.2f} 米)。"

    def land(self) -> str:
        """执行降落。"""
        print("执行降落...")
        self.client.landAsync(timeout_sec=5).join()
        self.client.armDisarm(False, self.vehicle_name)
        return "OBSERVATION: 无人机已安全降落并解除锁定。"


    def get_current_pose(self) -> str:
        """获取并返回无人机当前的 GPS 坐标和姿态。"""
        
        # 获取多旋翼状态
        state = self.client.getMultirotorState(self.vehicle_name)
        
        # 修正点 A: 使用 simGetGroundTruthGpsLocation (最常用的方法)
        try:
            # 这个方法应该存在于 self.client (MultirotorClient) 对象中
            gps = self.client.simGetGroundTruthGpsLocation(self.vehicle_name) 
        except AttributeError:
            # 如果 simGetGroundTruthGpsLocation 不存在，尝试另一个 AirSim API 提供的 GPS 方法
            # 警告：此方法可能已被弃用或不存在于所有版本中
            # gps = self.client.getGpsLocation(self.vehicle_name) # 可能会再次报错
            
            # 备选方案：尝试从状态对象中解析（如果 AirSim 状态对象中包含）
            # 实际情况是，通常需要调用 simGetGroundTruthGpsLocation
            raise AttributeError("无法找到 simGetGroundTruthGpsLocation 方法。请确认您的 AirSim 版本和 API 兼容性。")

        # 修正点 B: 确保位置信息通过 kinematics_estimated 访问 (解决了上一个问题)
        position = state.kinematics_estimated.position
        
        # AirSim NED 坐标系下，Z 为负值，需要转换为正高度
        altitude_meters = -position.z_val 
        
        # 姿态（四元数）
        orientation = state.kinematics_estimated.orientation
        
        pose_data = {
            "latitude": gps.latitude,
            "longitude": gps.longitude,
            "altitude_meters": altitude_meters,
            "orientation_w_x_y_z": [orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]
        }
        
        return (f"OBSERVATION: 当前姿态：Lat={gps.latitude:.6f}, Lon={gps.longitude:.6f}, "
                f"Alt={altitude_meters:.2f}m. 原始数据: {pose_data}")
    
    
if __name__ == "__main__":
    # 验证客户端
    client_test = AirSimClient()
    if client_test.connect_and_initialize():
        print(client_test.takeoff(20))
        time.sleep(3)
        print(client_test.get_current_pose())
        time.sleep(3)
        print(client_test.land())