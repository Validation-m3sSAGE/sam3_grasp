import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from piper_sdk import C_PiperInterface_V2

class PiperArmControllerSDK:
    """
    基于官方 Piper SDK 的机械臂控制器，用于替换 ROS2 版本。
    完全对齐原有 ROS 接口，支持上层平滑切换而无需修改算法逻辑。
    """
    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    GRIPPER_OPEN  = 0.085
    GRIPPER_CLOSE = 0.00
    HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]

    def __init__(self, can_port="can0"):
        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()
        
        self.is_enabled = False
        self._last_gripper_cmd = self.HOME_JOINTS[6]
        
        print(f"[SDK] Piper CAN 总线 ({can_port}) 初始化完成，等待底层状态反馈...")
        time.sleep(1.0)
        
    def enable_arm(self, enable: bool = True, wait: float = 1.0):
        if enable:
            while not self.piper.EnablePiper():
                time.sleep(0.01)
        else:
            while self.piper.DisablePiper():
                time.sleep(0.01)
        self.is_enabled = enable
        time.sleep(wait)
        print(f"[SDK] 硬件使能状态已设置为: {enable}")

    def set_joints(self, target_joints, speed_pct: int = 15, gripper_effort: float = 1.0, duration: float = 2.0, wait: bool = True) -> bool:
        if not self.is_enabled:
            self.enable_arm(True)

        speed_pct = int(np.clip(speed_pct, 1, 100))
        factor = 57295.7795  # 1000 * 180 / PI
        
        j0 = round(target_joints[0] * factor)
        j1 = round(target_joints[1] * factor)
        j2 = round(target_joints[2] * factor)
        j3 = round(target_joints[3] * factor)
        j4 = round(target_joints[4] * factor)
        j5 = round(target_joints[5] * factor)
        
        gripper_val = target_joints[6] if len(target_joints) >= 7 else self._last_gripper_cmd
        self._last_gripper_cmd = float(gripper_val)
        gripper_cmd = round(abs(gripper_val) * 1000000.0)

        # 下发关节(J)模式指令 0x01
        self.piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
        self.piper.JointCtrl(j0, j1, j2, j3, j4, j5)
        self.piper.GripperCtrl(gripper_cmd, 1000, 0x01, 0)
        
        if wait:
            time.sleep(duration)
        return True

    def go_home(self, speed_pct: int = 15, duration: float = 3.0) -> bool:
        print('[SDK] 执行硬件回零位...')
        return self.set_joints(self.HOME_JOINTS, speed_pct=speed_pct, duration=duration)

    def set_gripper(self, position: float, effort: float = 1.0, duration: float = 1.0) -> bool:
        position = float(position) if position >= 0.0 else 0.0
        self._last_gripper_cmd = position
        gripper_cmd = round(position * 1000000.0)
        self.piper.GripperCtrl(gripper_cmd, 1000, 0x01, 0)
        time.sleep(duration)
        return True

    def set_pose(self, x: float, y: float, z: float, roll: float = 0.0, pitch: float = 1.5708, yaw: float = 0.0, gripper: float = None, wait: bool = True, duration: float = 2.0) -> bool:
        if not self.is_enabled:
            self.enable_arm(True)

        if gripper is None:
            gripper = self._last_gripper_cmd
        else:
            gripper = float(gripper) if gripper >= 0.0 else 0.0
            self._last_gripper_cmd = gripper

        # 国际单位制映射至 SDK 微单位制
        factor_len = 1000000.0
        factor_deg = 57295.7795
        
        X = round(x * factor_len)
        Y = round(y * factor_len)
        Z = round(z * factor_len)
        RX = round(roll * factor_deg)
        RY = round(pitch * factor_deg)
        RZ = round(yaw * factor_deg)

        # 切换至笛卡尔(P)控制模式 0x00，使用平滑速度 50
        self.piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        
        gripper_cmd = round(gripper * factor_len)
        self.piper.GripperCtrl(gripper_cmd, 1000, 0x01, 0)
        
        if wait:
            time.sleep(duration)
            T_curr = self.get_current_pose_matrix()
            curr_x, curr_y, curr_z = T_curr[:3, 3]
            pos_error = np.sqrt((curr_x - x)**2 + (curr_y - y)**2 + (curr_z - z)**2)
            if pos_error > 0.03:
                print(f"[SDK Error] 运动学异常拦截: 目标({x:.3f}, {y:.3f}, {z:.3f}), "
                      f"实际滞留点({curr_x:.3f}, {curr_y:.3f}, {curr_z:.3f}), 误差 {pos_error:.3f}m")
                return False
        return True

    def get_current_pose_matrix(self) -> np.ndarray:
        msg = self.piper.GetArmEndPoseMsgs().end_pose
        # SDK 逆向解算，还原标准米与弧度
        x = msg.X_axis / 1000000.0
        y = msg.Y_axis / 1000000.0
        z = msg.Z_axis / 1000000.0
        rx = math.radians(msg.RX_axis / 1000.0)
        ry = math.radians(msg.RY_axis / 1000.0)
        rz = math.radians(msg.RZ_axis / 1000.0)
        
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        T[:3, :3] = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        return T

    def wait_for_pose(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                return False
            # 轮询 SDK 后台是否接收到有效的高频反馈帧
            if self.piper.GetArmEndPoseMsgs().Hz > 0:
                return True
            time.sleep(0.05)
            
    @property
    def current_joints(self):
        msg = self.piper.GetArmJointMsgs().joint_state
        factor = 57295.7795
        j0 = msg.joint_1 / factor
        j1 = msg.joint_2 / factor
        j2 = msg.joint_3 / factor
        j3 = msg.joint_4 / factor
        j4 = msg.joint_5 / factor
        j5 = msg.joint_6 / factor
        return np.array([j0, j1, j2, j3, j4, j5, self._last_gripper_cmd])

    def destroy_node(self):
        # 兼容 ROS 框架的析构调用
        self.piper.DisconnectPort()
