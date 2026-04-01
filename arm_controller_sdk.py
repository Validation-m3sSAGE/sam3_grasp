import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from piper_sdk import C_PiperInterface_V2

import os
import numpy as np
from ompl_planner import OMPLKinematicsPlanner

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
        
        urdf_path = os.path.join(os.path.dirname(__file__), "urdf", "piper_description.urdf")
        self.planner = OMPLKinematicsPlanner(urdf_path)
        
        print(f"[SDK] Piper CAN 总线 ({can_port}) 初始化完成，OMPL 物理引擎已挂载...")
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
        """
        全量接管为 OMPL/关节空间轨迹规划:
        将 Cartesian 目标坐标经过本地 URDF 物理引擎离线逆解，映射为无奇异的高维关节角度流形，
        并调用 set_joints 进行安全插补，彻底规避硬件固件中的笛卡尔坐标越界报错。
        """
        if not self.is_enabled:
            self.enable_arm(True)

        if gripper is None:
            gripper = self._last_gripper_cmd
        else:
            gripper = float(gripper) if gripper >= 0.0 else 0.0
            self._last_gripper_cmd = gripper

        try:
            current_joints = self.current_joints[:6]
            target_joints = self.planner.solve_ik(
                target_pos=[x, y, z],
                target_rpy=[roll, pitch, yaw],
                current_joints=current_joints
            )
        except Exception as e:
            print(f"[Planner Error] OMPL/IK 离线引擎解算失败: {e}")
            return False

        full_target_joints = np.append(target_joints, gripper)
        
        print(f"[Planner] 坐标投影成功 ({x:.3f}, {y:.3f}, {z:.3f}) -> 映射关节域: {np.round(target_joints, 2)}")
        
        # 抛弃固件层的 EndPoseCtrl 笛卡尔模式，依赖可靠的关节驱动域完成物理移动
        return self.set_joints(full_target_joints, speed_pct=25, duration=duration, wait=wait)

    def get_current_pose_matrix(self) -> np.ndarray:
        """使用 OMPL/PyBullet 物理引擎进行纯数学正向解算，与 IK 绝对对齐"""
        if hasattr(self, 'planner'):
            current_joints = self.current_joints[:6]
            return self.planner.solve_fk(current_joints)
        else:
            return np.eye(4)

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
        if hasattr(self, 'planner'):
            self.planner.destroy()
        self.piper.DisconnectPort()

if __name__ == "__main__":
    pas = PiperArmControllerSDK()
    pas.set_gripper(0.9)
