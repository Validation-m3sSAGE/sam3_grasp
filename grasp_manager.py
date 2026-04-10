import numpy as np
import time
import cv2

import os
from perception_layer import PerceptionLayer
from arm_controller_sdk import PiperArmControllerSDK

# 机械臂夹爪中心在相机坐标系下的近似偏置 (X, Y, Z 单位: 米)
GRIPPER_OFFSET_IN_CAM = np.array([0.02, 0.05, 0.10])

# --- 夹爪物理末端参数配置 (可根据真实硬件动态微调) ---
TCP_TOOL_LEN = 0.075       # [修正] 法兰盘面到夹爪尖端的真实物理长度 (基于您前事实测的7.5cm)
GRASP_PENETRATION = 0.020  # [微调] 随短夹爪同比例下调的刺入预期，避免挤压形变

class GraspManager:
    def go_standby(self, wait=True, duration=3.0):
        """替代传统回零，强制系统回归统一的斜向下观测位姿，消灭任何视角的折叠切换"""
        print("[Standby] 移动至标准斜向待机位姿 (X=0.25, Y=0, Z=0.32, Pitch=135°)...")
        return self.arm.set_pose(0.25, 0.0, 0.32, pitch=2.35619, yaw=0.0, wait=wait, duration=duration)

    def __init__(self, calib_path=None):
        self.arm = PiperArmControllerSDK(can_port="can0")

        time.sleep(0.5)
        self.perception = PerceptionLayer()
        self.calib_path = calib_path or "calibration_result.npz"
        self.T_ee_to_cam = None
        
        # 启动时默认尝试加载历史标定文件
        if os.path.exists(self.calib_path):
            try:
                data = np.load(self.calib_path)
                if 'T_ee_to_cam' in data:
                    T = data['T_ee_to_cam']
                    translation = T[:3, 3]
                    # 健全性检查: 任何轴的物理偏移都不应超过 15cm
                    if np.any(np.abs(translation) > 0.15):
                        print(f"[Init] [警告!] 加载的历史手眼矩阵平移分量异常 {np.round(translation, 4)}，可能已坍缩。")
                        print(f"[Init]           将忽略此文件，请必须重新执行手眼标定！")
                        self.T_ee_to_cam = None
                    else:
                        self.T_ee_to_cam = T
                        print(f"[Init] 成功加载历史手眼装配矩阵: {self.calib_path}")
                else:
                    print("[Init] 发现被弃用的旧版标定数据 (仅包含旋转矩阵)，请重新执行手眼标定。")
            except Exception as e:
                print(f"[Init] 历史标定文件读取失败 ({e})，请稍后手动执行标定。")

        print("[Init] 系统初始化完成，开始执行俯视待机动作...")
        self.go_standby()

    def calibrate_axes(self, pattern_size=(11, 8), square_size=0.01, num_poses=15):
        """真正的多位姿视觉手眼标定 (Tsai-Lenz, 眼在手上)"""
        print(f"\n[Calibration] 启动标准 AX=XB 手眼标定流程...")
        print(f"[Calibration] 参数: 棋盘格 9x12 格子 -> 内角点 {pattern_size[0]}x{pattern_size[1]}, 物理边长 {square_size} 米。")
        print("[Calibration] ⚠️ 请确保此时标定板平放在下方正中位置。")
        
        self.arm.enable_arm(True)
        
        print("[Calibration] 强制移动至标准斜向观测姿态作为标定中心...")
        self.arm.set_pose(0.25, 0.0, 0.32, pitch=2.35619, yaw=0.0, wait=True, duration=3.0)
        time.sleep(1.0)
        
        # 使用当前俯视关节角作为复位锚点
        base_joints = self.arm.current_joints[:6]
        
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        import random
        from scipy.spatial.transform import Rotation as R_scipy
        
        valid_poses = 0
        
        # 强制锁定理论中心进行平移与姿态的常数基准 (改用135度pitch以绝对规避手腕截断)
        base_x, base_y, base_z = 0.25, 0.0, 0.32
        base_roll, base_pitch, base_yaw = 0.0, 2.35619, 0.0
        
        for i in range(num_poses):
            print(f"\n[Calibration] 正在自动采集第 {i+1}/{num_poses} 个观测位姿...")
            
            # 首次原位采集，后续施加安全的笛卡尔空间微扰
            if i > 0:
                pose_reached = False
                attempts = 0
                while not pose_reached and attempts < 5:
                    # 为确保求解器收敛，平移扰动应更小，旋转扰动应更大
                    dx = random.uniform(-0.03, 0.03)
                    dy = random.uniform(-0.03, 0.03)
                    dz = random.uniform(-0.02, 0.02)
                    
                    # 姿态扰动扩展至 ±0.35 弧度 (约 20度)，为 AX=XB 提供充分的旋转信息
                    droll = random.uniform(-0.35, 0.35)
                    dpitch = random.uniform(-0.35, 0.35)
                    dyaw = random.uniform(-0.35, 0.35)
                    
                    # 直接在常数基准上进行代数叠加，杜绝库间矩阵转换引发的奇异构型别名
                    pose_reached = self.arm.set_pose(base_x + dx, base_y + dy, base_z + dz, 
                                      roll=base_roll + droll, 
                                      pitch=base_pitch + dpitch, 
                                      yaw=base_yaw + dyaw, 
                                      wait=True, duration=2.5)
                    attempts += 1
                
                if not pose_reached:
                    print("[Calibration] 警告: 此随机扰动位姿处于工作空间外，跳过。")
                    continue
            else:
                self.arm.set_joints(np.append(base_joints, self.arm._last_gripper_cmd), speed_pct=20, wait=True, duration=2.5)
                
            time.sleep(1.0)
            
            # 1. 解算棋盘格在相机系中的 6D 位姿
            success, T_cam2board = self.perception.get_chessboard_pose(pattern_size, square_size)
            if not success:
                print("[Calibration] ❌ 未能提取完整的棋盘格角点，跳过此位姿。")
                continue
                
            # 2. 读取当前机械臂底层的正运动学末端矩阵
            T_base2ee = self.arm.get_current_pose_matrix()
            
            R_g2b = T_base2ee[:3, :3]
            t_g2b = T_base2ee[:3, 3]
            R_t2c = T_cam2board[:3, :3]
            t_t2c = T_cam2board[:3, 3]
            
            R_gripper2base.append(R_g2b)
            t_gripper2base.append(t_g2b)
            R_target2cam.append(R_t2c)
            t_target2cam.append(t_t2c)
            valid_poses += 1
            print(f"[Calibration] ✅ 角点已对齐！目前有效数据: {valid_poses} 组。")

        print("\n[Calibration] 自动采集结束，机械臂正在复位...")
        self.arm.set_joints(np.append(base_joints, self.arm._last_gripper_cmd), speed_pct=20, wait=True, duration=2.5)
        
        if valid_poses < 5:
            raise RuntimeError(f"有效数据极度匮乏 (仅 {valid_poses} 组，需至少 5 组)。请拉高相机视野，确保扰动时标定板不会出画。")
            
        print("[Calibration] 开始执行 Park & Martin 非线性 AX=XB 矩阵逼近求解 (平移鲁棒性更强)...")
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, 
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK
        )
        
        T_ee_to_cam = np.eye(4)
        T_ee_to_cam[:3, :3] = R_cam2gripper
        T_ee_to_cam[:3, 3] = t_cam2gripper.flatten()
        
        print("\n[Calibration] 标定彻底大功告成！常量装配矩阵 T_ee_to_cam (末端法兰到相机光心):")
        print(np.round(T_ee_to_cam, 4))
        
        self.T_ee_to_cam = T_ee_to_cam
        np.savez(self.calib_path, T_ee_to_cam=self.T_ee_to_cam)
        print(f"[Calibration] 常量矩阵已成功持久化至硬盘: {self.calib_path}")
        
        return T_ee_to_cam

    def check_target_in_scene(self, target_name='apple'):
        """检查目标物体是否存在于当前视野中"""
        print(f"[Check Target] 开始检测目标: {target_name}")
        result = self.perception.process(target_name, visualize=False)
        if result['success']:
            print(f"[Check Target] 目标 '{target_name}' 已在视野中找到。")
            return {'success': True, 'found': True}
        else:
            print(f"[Check Target] 目标 '{target_name}' 未找到 ({result.get('error', '')})。")
            return {'success': True, 'found': False}

    def grasp_simple(self, target_name='apple', auto_home=True, skip_observation_pose=False):
        """单次直接抓取执行管线"""
        print(f"[Grasp Simple] 开始执行单次抓取，目标: {target_name}")
        self.arm.enable_arm(True)
        self.arm.set_gripper(self.arm.GRIPPER_OPEN)

        if self.T_ee_to_cam is None:
            print("[Error] 缺失装配常数矩阵且未加载到历史文件，请先执行手眼标定。")
            return {'success': False, 'error': 'Missing calibration matrix'}

        if not skip_observation_pose:
            print("[Grasp Simple] 移动至初始斜向观测位姿...")
            self.arm.set_pose(0.25, 0.0, 0.32, pitch=2.35619, yaw=0.0, wait=True, duration=2.5)
            time.sleep(1.0)
        else:
            print("[Grasp Simple] 已跳过初始复位，直接在当前已对准的视角下执行后续抓取。")

        result = self.perception.process(target_name, visualize=False)
        if not result['success']:
            error_msg = result.get('error', '未知异常')
            print(f"[Grasp Simple] 错误: 感知管线执行失败 ({error_msg})")
            return {'success': False, 'action_completed': False}

        p_target_cam_3d = result['grasp_point_cam']
        print(f"[Grasp Simple] 目标 3D 坐标 (相机局部): {np.round(p_target_cam_3d, 4)} m")

        # === 核心坐标轴自洽诊断打印 ===
        P_cam = np.array([p_target_cam_3d[0], p_target_cam_3d[1], p_target_cam_3d[2], 1.0])
        T_base_to_ee = self.arm.get_current_pose_matrix()
        
        print("\n" + "="*60)
        print("[Diagnostic] 坐标系映射矩阵物理自洽校验")
        print("="*60)
        
        print(f"1. 目标局部坐标 (相机系 P_cam):")
        print(f"   X_c={P_cam[0]:.4f}, Y_c={P_cam[1]:.4f}, Z_c={P_cam[2]:.4f} m")
        
        # 提取手眼矩阵的平移分量（相机相对于末端法兰的物理安装偏移）
        cam_offset_x, cam_offset_y, cam_offset_z = self.T_ee_to_cam[:3, 3]
        print(f"\n2. 手眼装配矩阵平移量 (T_ee_to_cam):")
        print(f"   dx={cam_offset_x:.4f}m, dy={cam_offset_y:.4f}m, dz={cam_offset_z:.4f}m")
        print(f"   -> [常识判断] 此值代表相机光心到法兰中心的物理距离。若某轴绝对值 > 0.15m，说明标定平移已坍缩！")
        
        # 提取当前法兰在基座系的绝对坐标
        ee_x, ee_y, ee_z = T_base_to_ee[:3, 3]
        print(f"\n3. 当前法兰绝对坐标 (T_base_to_ee 正运动学):")
        print(f"   X_f={ee_x:.4f}, Y_f={ee_y:.4f}, Z_f={ee_z:.4f} m")
        
        # 计算相机的绝对物理坐标
        T_base_to_cam = T_base_to_ee @ self.T_ee_to_cam
        cam_x, cam_y, cam_z = T_base_to_cam[:3, 3]
        print(f"\n4. 当前相机绝对物理坐标 (T_base_to_cam 推导值):")
        print(f"   X_cam={cam_x:.4f}, Y_cam={cam_y:.4f}, Z_cam={cam_z:.4f} m")
        
        P_base = T_base_to_cam @ P_cam
        target_x, target_y, target_z = P_base[:3]
        print(f"\n5. 最终目标绝对坐标 (P_base):")
        print(f"   X_t={target_x:.4f}, Y_t={target_y:.4f}, Z_t={target_z:.4f} m")
        print("="*60 + "\n")

        # 物理工作空间提示
        target_radius = np.hypot(target_x, target_y)
        if target_radius > 0.45 or target_z < -0.05 or target_z > 0.40:
            print(f"[Warning] 目标点可能超限！距基座平距 {target_radius:.3f}m, 高度 {target_z:.3f}m。")

        # 引入感知层计算的物体偏航角，使夹爪完美对齐物体长轴
        raw_yaw = result.get('yaw', 0.0)
        # 将 yaw 归一化到 [-pi/2, pi/2] 以防超出 Joint 6 (-2.09~2.09) 的物理限位
        while raw_yaw > 1.5708: raw_yaw -= 3.14159
        while raw_yaw < -1.5708: raw_yaw += 3.14159
        grasp_yaw = raw_yaw

        import math
        # --- 强制写死 45度斜向下抓取 ---
        target_radius = np.hypot(target_x, target_y)
        
        # 放宽安全半径至 0.46m。由于底层 OMPL 规划器已经具备完善的运动学和关节限位阻断，
        # 我们不需要在这里做过于保守的软拦截，直接交由 IK 引擎判断是否可达。
        if target_radius > 0.66:
            print(f"[Error] 目标距离 ({target_radius:.3f}m) 超过 0.66m 的绝对物理臂展，已放弃抓取。")
            return {'success': False, 'error': 'Target too far, grasp aborted'}
            
        effective_retract = TCP_TOOL_LEN - GRASP_PENETRATION
        
        # 固定下倾角 45度 (0.7854 rad)
        alpha_rad = 0.785398
        grasp_pitch = 1.5708 + alpha_rad  # 135度
        
        # 偏航角：强行写死为基座指向目标的夹角，直接正对目标过去
        grasp_yaw = math.atan2(target_y, target_x)
        
        print(f"[Action] 执行固定 45度斜向下抓取，目标平距 {target_radius:.3f}m。")
        
        vec_x = target_x / target_radius if target_radius > 0.001 else 1.0
        vec_y = target_y / target_radius if target_radius > 0.001 else 0.0
        
        # 计算 45度下应有的物理退距分量
        dx_plane = effective_retract * math.cos(alpha_rad)
        dz = effective_retract * math.sin(alpha_rad)
        
        # --- 引入战术级纯水平后撤补偿 (Pull-back Offset) ---
        # 吸收 URDF 连杆原点与真实夹爪安装面之间的隐性距离差，
        # 让整个夹爪在 XY 平面上向基座回缩 3.5 厘米，避免把苹果吞到夹爪根部
        PULL_BACK = 0.05
        dx_plane += PULL_BACK
        print(f"[Action] 触发水平战术后撤补偿，将抓取点向回拉缩 {PULL_BACK * 1000}mm 以对齐指尖。")
        
        # --- 引入横向对齐补偿 (Lateral Offset) ---
        # 针对夹爪稳定偏左的情况，计算与射向垂直的“向右”法向量
        # 单位前向向量为 (vec_x, vec_y)，顺时针旋转 90 度的右侧向量为 (vec_y, -vec_x)
        LATERAL_OFFSET = 0.025
        lat_dx = LATERAL_OFFSET * vec_y
        lat_dy = LATERAL_OFFSET * (-vec_x)
        print(f"[Action] 触发横向对齐补偿，将抓取点整体向右平移 {LATERAL_OFFSET * 1000}mm。")
        
        flange_grasp_x = target_x - dx_plane * vec_x + lat_dx
        flange_grasp_y = target_y - dx_plane * vec_y + lat_dy
        flange_grasp_z = target_z + dz
        
        # 严格防撞底线：45度姿态下，张开的夹爪下半指宽(4.25cm)会斜向刺向桌面
        # 计算理论最低点 = 目标 Z - 刺入深度*sin(45) - 夹爪半宽*cos(45)
        lowest_z = target_z - GRASP_PENETRATION * math.sin(alpha_rad) - 0.0425 * math.cos(alpha_rad)
        if lowest_z < 0.005:
            lift_amount = 0.005 - lowest_z
            flange_grasp_z += lift_amount
            print(f"[Warning] 45度预期姿态夹爪下缘可能击穿桌面！已强制将法兰垂直抬高 {lift_amount*1000:.1f}mm 进行避让。")

        # 法兰底线：额外保护电机硬件
        if flange_grasp_z < 0.065:
            flange_grasp_z = 0.065
            print("[Warning] 法兰高度低于 65mm 安全底线，已执行兜底抬升。")
        
        # 预抓取点：沿相同 45度 倾角轨迹向斜后方退 10 厘米
        pre_dx = 0.10 * math.cos(alpha_rad)
        pre_dz = 0.10 * math.sin(alpha_rad)
        flange_pre_x = flange_grasp_x - pre_dx * vec_x
        flange_pre_y = flange_grasp_y - pre_dx * vec_y
        flange_pre_z = flange_grasp_z + pre_dz

        print(f"[Action] 移动至预抓取点 (法兰高度 {flange_pre_z:.3f}m, Pitch={math.degrees(grasp_pitch):.1f}°, Yaw={math.degrees(grasp_yaw):.1f}°)...")
        success = self.arm.set_pose(flange_pre_x, flange_pre_y, flange_pre_z, pitch=grasp_pitch, yaw=grasp_yaw, wait=True, duration=2.5)
        if not success:
            print("[Error] 预抓取点位姿不可达！")
            return {'success': False, 'error': 'Pre-grasp pose unreachable'}

        print(f"[Action] 沿连续倾角向目标推进 (法兰高度 {flange_grasp_z:.3f}m)...")
        success = self.arm.set_pose(flange_grasp_x, flange_grasp_y, flange_grasp_z, pitch=grasp_pitch, yaw=grasp_yaw, wait=True, duration=1.5)
        if not success:
            print("[Error] 抓取点位姿不可达！")
            return {'success': False, 'error': 'Grasp pose unreachable'}

        print("[Action] 闭合夹爪 (抓取力度 0.5)...")
        self.arm.set_gripper(self.arm.GRIPPER_CLOSE, effort=0.5, duration=1.5)

        print("[Action] 保持姿态沿原轨迹拔起...")
        self.arm.set_pose(flange_pre_x, flange_pre_y, flange_pre_z + 0.05, pitch=grasp_pitch, yaw=grasp_yaw, wait=True, duration=2.0)

        if auto_home:
            print("[Action] 抓取完成，自动回归标准俯视待机位姿...")
            self.go_standby()
        else:
            print("[Action] 抓取完成，保留在当前高度与平视姿态等待后续探索指令。")

        print("[Grasp Simple] 抓取动作链执行完毕。")
        return {'success': True, 'action_completed': True}

    def explore_and_place(self, target_name='trash can'):
        """旋转探索环境并在目标上方松开夹爪释放物体"""
        print(f"[Explore] 开始探索目标: {target_name}")
        
        # 先回到手眼标定的斜向初始观测位姿
        init_x, init_y, init_z = 0.25, 0.0, 0.32
        print(f"[Explore] 回到斜向标定初始位置 ({init_x}, {init_y}, {init_z})...")
        self.arm.set_pose(init_x, init_y, init_z, pitch=2.35619, yaw=0.0, wait=True, duration=2.5)
        time.sleep(1.0)
        
        curr_x, curr_y, curr_z = init_x, init_y, init_z
        
        current_yaw = 0.0
        
        # 放置时的偏置调整: 相机系下Z为正前方，Y为向下。
        place_offset_cam = np.array([0.0, -0.15, 0.05]) 
            
        found = False
        target_base_pos = None
        
        step_angle = 0.5236
        total_steps = 12
        base_joints = self.arm.current_joints.copy()
        
        import math
        for step in range(total_steps):
            if step > 0:
                current_yaw += step_angle
                print(f"[Explore] 机械臂旋转探索... 当前 Yaw: {np.degrees(current_yaw):.1f}°")
                rotate_joints = base_joints.copy()
                rotate_joints[0] = current_yaw
                self.arm.set_joints(rotate_joints, speed_pct=15, duration=2.0)
                time.sleep(1.0)
                
            # 执行目标分割检查
            result = self.perception.process(target_name, visualize=False)
            if result['success']:
                print(f"[Explore] 识别到目标实体 '{target_name}'，启动画面居中对准程序...")
                
                align_attempts = 0
                while align_attempts < 5:
                    p_target_cam_3d = result['grasp_point_cam']
                    cam_dx = p_target_cam_3d[0]
                    cam_z = p_target_cam_3d[2]
                    
                    # 若目标在相机横向偏离小于 1.5cm，认为已基本居中
                    if abs(cam_dx) < 0.015:
                        print(f"[Explore] 目标已对准画面中心 (横向偏移 {cam_dx * 1000:.1f}mm)。")
                        break
                        
                    # 计算需要补偿的底座偏航角
                    # 引入比例控制系数 0.5 进行阻尼衰减，抵消相机旋转半径过大引发的过调震荡
                    comp_yaw = math.atan2(-cam_dx, cam_z) * 0.5
                    current_yaw += comp_yaw
                    
                    print(f"[Explore] 目标偏离中心 {cam_dx * 1000:.1f}mm，微调底座 Yaw 角度 {math.degrees(comp_yaw):.1f}° (已阻尼衰减)...")
                    rotate_joints = base_joints.copy()
                    rotate_joints[0] = current_yaw
                    self.arm.set_joints(rotate_joints, speed_pct=10, duration=1.5)
                    time.sleep(1.0)
                    
                    result = self.perception.process(target_name, visualize=False)
                    if not result['success']:
                        print("[Explore] 警告: 微调对准过程中丢失目标。")
                        break
                    align_attempts += 1
                
                # 经过对准后，必须确保结果依然成功才能进行几何解算
                if result.get('success', False):
                    p_target_cam_3d = result['grasp_point_cam']
                    
                    # 纯粹的几何映射，求出垃圾桶的物理绝对坐标
                    P_cam = np.array([p_target_cam_3d[0], p_target_cam_3d[1], p_target_cam_3d[2], 1.0])
                    
                    T_base_to_ee = self.arm.get_current_pose_matrix()
                    T_base_to_cam = T_base_to_ee @ self.T_ee_to_cam
                    P_base = T_base_to_cam @ P_cam
                    
                    target_base_pos = P_base[:3]
                    found = True
                    break
                
        if not found:
            print(f"[Explore] 探索周期结束，视野内未匹配 '{target_name}'，回归标准俯视位姿。")
            self.go_standby()
            return {'success': False, 'error': '未找到投放目标'}
            
        tx, ty, tz = target_base_pos
        print(f"[Explore] 垃圾桶物理中心坐标: X={tx:.4f}, Y={ty:.4f}, Z={tz:.4f}")
        
        target_radius = np.hypot(tx, ty)
        
        # 将放置任务的安全平距截断大幅放宽至 0.46m，让手臂尽力向外探出
        if target_radius > 0.46:
            scale = 0.46 / target_radius
            tx = tx * scale
            ty = ty * scale
            print(f"[Explore] 垃圾桶中心距离过远 ({target_radius:.3f}m)，已沿射线方向极限向外探出至法兰平距 0.46m。")
            
        # 释放目标点：法兰高度需要考虑到垃圾桶的高度
        drop_flange_z = max(tz + TCP_TOOL_LEN + 0.15, 0.30)
        print(f"[Explore] 准备在目标上方高空释放 (法兰高度 Z={drop_flange_z:.4f})")
        
        import math
        # 释放姿态优化：放弃容易引起手腕死锁的纯垂直向下(180度)，改为 135度(斜向下45度) 舒适姿态
        place_pitch = 2.35619 
        place_yaw = math.atan2(ty, tx)

        print(f"[Action] 尝试直接移动至释放点 (X={tx:.3f}, Y={ty:.3f}, Z={drop_flange_z:.3f}, Pitch=135°, Yaw={math.degrees(place_yaw):.0f}°)...")
        success = self.arm.set_pose(tx, ty, drop_flange_z, pitch=place_pitch, yaw=place_yaw, wait=True, duration=2.5)
        
        if success:
            print("[Action] 已到达目标正上方，松开夹爪释放物体...")
            self.arm.set_gripper(self.arm.GRIPPER_OPEN, duration=1.5)
            task_success = True
        else:
            print("[Error] 目标释放点不可达！系统拒绝松开夹爪，当前物体依然处于夹持保护状态。")
            task_success = False
            
        print("[Action] 动作执行完毕，机械臂带着当前夹持状态回归标准待机位姿...")
        self.go_standby()
        
        return {'success': task_success}

    def explore_and_grasp(self, target_name='apple'):
        """旋转向右探索寻找目标，一旦找到即执行画面居中对准，并直接下探抓取"""
        print(f"[Explore Grasp] 开始探索抓取目标: {target_name}")
        
        self.arm.enable_arm(True)
        print("[Explore Grasp] 张开夹爪准备抓取...")
        self.arm.set_gripper(self.arm.GRIPPER_OPEN)
        
        print(f"[Explore Grasp] 移动至斜向初始位置进行首次视觉检查...")
        self.arm.set_pose(0.25, 0.0, 0.32, pitch=2.35619, yaw=0.0, wait=True, duration=2.5)
        time.sleep(1.0)
        
        current_yaw = 0.5236*3
        # 强制向右转：底座减去固定弧度 (0.5236rad 约 30度)
        step_angle = -0.5236 
        total_steps = 12
        base_joints = self.arm.current_joints.copy()
        rotate_joints = base_joints.copy()
        rotate_joints[0] = current_yaw
        self.arm.set_joints(rotate_joints, speed_pct=15, duration=2.0)
        time.sleep(2.0)
        found = False
        import math
        
        for step in range(total_steps):
            if step > 0:
                current_yaw += step_angle
                print(f"[Explore Grasp] 视野未见目标，继续向右旋转探索... 当前 Yaw: {np.degrees(current_yaw):.1f}°")
                rotate_joints = base_joints.copy()
                rotate_joints[0] = current_yaw
                self.arm.set_joints(rotate_joints, speed_pct=15, duration=2.0)
                time.sleep(1.0)
                
            result = self.perception.process(target_name, visualize=False)
            if result['success']:
                print(f"[Explore Grasp] 识别到目标实体 '{target_name}'，启动画面居中对准程序...")
                
                align_attempts = 0
                while align_attempts < 5:
                    cam_dx = result['grasp_point_cam'][0]
                    cam_z = result['grasp_point_cam'][2]
                    
                    if abs(cam_dx) < 0.015:
                        print(f"[Explore Grasp] 目标已对准画面中心 (横向偏移 {cam_dx * 1000:.1f}mm)。")
                        break
                        
                    # 引入比例控制系数 0.5 进行阻尼衰减，抵消相机旋转半径过大引发的过调震荡
                    comp_yaw = math.atan2(-cam_dx, cam_z) * 0.5
                    current_yaw += comp_yaw
                    
                    print(f"[Explore Grasp] 目标偏离中心 {cam_dx * 1000:.1f}mm，微调底座 Yaw 角度 {math.degrees(comp_yaw):.1f}° (已阻尼衰减)...")
                    rotate_joints = base_joints.copy()
                    rotate_joints[0] = current_yaw
                    self.arm.set_joints(rotate_joints, speed_pct=10, duration=1.5)
                    time.sleep(1.0)
                    
                    result = self.perception.process(target_name, visualize=False)
                    if not result['success']:
                        print("[Explore Grasp] 警告: 微调对准过程中丢失目标。")
                        break
                    align_attempts += 1
                
                if result.get('success', False):
                    found = True
                    break
                    
        if not found:
            print(f"[Explore Grasp] 探索周期结束，未找到目标 '{target_name}'，回收回归待机位。")
            self.go_standby()
            return {'success': False, 'error': '未找到目标'}
            
        print(f"[Explore Grasp] 目标已稳定对准，将直接移交单次抓取管线执行动作...")
        # 传递 skip_observation_pose=True，让 grasp_simple 沿用当前已经对准的角度和位姿
        return self.grasp_simple(target_name=target_name, auto_home=True, skip_observation_pose=True)

    def explore_right_and_place(self, target_name='trash can'):
        """向右旋转探索环境并在目标上方松开夹爪释放物体"""
        print(f"[Explore Right] 开始向右探索目标: {target_name}")
        
        # 先回到手眼标定的斜向初始观测位姿
        init_x, init_y, init_z = 0.25, 0.0, 0.32
        print(f"[Explore Right] 回到斜向标定初始位置 ({init_x}, {init_y}, {init_z})...")
        self.arm.set_pose(init_x, init_y, init_z, pitch=2.35619, yaw=0.0, wait=True, duration=2.5)
        time.sleep(1.0)
        
        curr_x, curr_y, curr_z = init_x, init_y, init_z
        current_yaw = 0.0
        
        found = False
        target_base_pos = None
        
        # 向右转为负角度 (-30度)
        step_angle = -0.5236
        total_steps = 12
        base_joints = self.arm.current_joints.copy()
        
        import math
        for step in range(total_steps):
            if step > 0:
                current_yaw += step_angle
                print(f"[Explore Right] 机械臂向右旋转探索... 当前 Yaw: {np.degrees(current_yaw):.1f}°")
                rotate_joints = base_joints.copy()
                rotate_joints[0] = current_yaw
                self.arm.set_joints(rotate_joints, speed_pct=15, duration=2.0)
                time.sleep(1.0)
                
            # 执行目标分割检查
            result = self.perception.process(target_name, visualize=False)
            if result['success']:
                print(f"[Explore Right] 识别到目标实体 '{target_name}'，启动画面居中对准程序...")
                
                align_attempts = 0
                while align_attempts < 5:
                    p_target_cam_3d = result['grasp_point_cam']
                    cam_dx = p_target_cam_3d[0]
                    cam_z = p_target_cam_3d[2]
                    
                    # 若目标在相机横向偏离小于 1.5cm，认为已基本居中
                    if abs(cam_dx) < 0.015:
                        print(f"[Explore Right] 目标已对准画面中心 (横向偏移 {cam_dx * 1000:.1f}mm)。")
                        break
                        
                    # 计算需要补偿的底座偏航角
                    # 引入比例控制系数 0.5 进行阻尼衰减，抵消相机旋转半径过大引发的过调震荡
                    comp_yaw = math.atan2(-cam_dx, cam_z) * 0.5
                    current_yaw += comp_yaw
                    
                    print(f"[Explore Right] 目标偏离中心 {cam_dx * 1000:.1f}mm，微调底座 Yaw 角度 {math.degrees(comp_yaw):.1f}° (已阻尼衰减)...")
                    rotate_joints = base_joints.copy()
                    rotate_joints[0] = current_yaw
                    self.arm.set_joints(rotate_joints, speed_pct=10, duration=1.5)
                    time.sleep(1.0)
                    
                    result = self.perception.process(target_name, visualize=False)
                    if not result['success']:
                        print("[Explore Right] 警告: 微调对准过程中丢失目标。")
                        break
                    align_attempts += 1
                
                # 经过对准后，必须确保结果依然成功才能进行几何解算
                if result.get('success', False):
                    p_target_cam_3d = result['grasp_point_cam']
                    
                    # 纯粹的几何映射，求出垃圾桶的物理绝对坐标
                    P_cam = np.array([p_target_cam_3d[0], p_target_cam_3d[1], p_target_cam_3d[2], 1.0])
                    
                    T_base_to_ee = self.arm.get_current_pose_matrix()
                    T_base_to_cam = T_base_to_ee @ self.T_ee_to_cam
                    P_base = T_base_to_cam @ P_cam
                    
                    target_base_pos = P_base[:3]
                    found = True
                    break
                
        if not found:
            print(f"[Explore Right] 探索周期结束，视野内未匹配 '{target_name}'，回归标准俯视位姿。")
            self.go_standby()
            return {'success': False, 'error': '未找到投放目标'}
            
        tx, ty, tz = target_base_pos
        print(f"[Explore Right] 目标物理中心坐标: X={tx:.4f}, Y={ty:.4f}, Z={tz:.4f}")
        
        target_radius = np.hypot(tx, ty)
        
        # 将放置任务的安全平距截断大幅放宽至 0.46m，让手臂尽力向外探出
        if target_radius > 0.46:
            scale = 0.46 / target_radius
            tx = tx * scale
            ty = ty * scale
            print(f"[Explore Right] 目标中心距离过远 ({target_radius:.3f}m)，已沿射线方向极限向外探出至法兰平距 0.46m。")
            
        # 释放目标点：法兰高度需要考虑到垃圾桶的高度
        drop_flange_z = max(tz + TCP_TOOL_LEN + 0.15, 0.30)
        print(f"[Explore Right] 准备在目标上方高空释放 (法兰高度 Z={drop_flange_z:.4f})")
        
        # 释放姿态优化：放弃容易引起手腕死锁的纯垂直向下(180度)，改为 135度(斜向下45度) 舒适姿态
        place_pitch = 2.35619 
        place_yaw = math.atan2(ty, tx)

        print(f"[Action] 尝试直接移动至释放点 (X={tx:.3f}, Y={ty:.3f}, Z={drop_flange_z:.3f}, Pitch=135°, Yaw={math.degrees(place_yaw):.0f}°)...")
        success = self.arm.set_pose(tx, ty, drop_flange_z, pitch=place_pitch, yaw=place_yaw, wait=True, duration=2.5)
        
        if success:
            print("[Action] 已到达目标正上方，松开夹爪释放物体...")
            self.arm.set_gripper(self.arm.GRIPPER_OPEN, duration=1.5)
            task_success = True
        else:
            print("[Error] 目标释放点不可达！系统拒绝松开夹爪，当前物体依然处于夹持保护状态。")
            task_success = False
            
        print("[Action] 动作执行完毕，机械臂带着当前夹持状态回归标准待机位姿...")
        self.go_standby()
        
        return {'success': task_success}

    def rotate_and_release(self, angle_deg=60.0):
        """旋转指定角度后释放夹持物体"""
        print(f"[Rotate Release] 旋转 {angle_deg}° 后释放...")
        self.arm.enable_arm(True)
        
        angle_rad = np.radians(angle_deg)
        
        # 先到斜向伸展位姿，避免从折叠构型旋转后末端贴近基座
        self.arm.set_pose(0.25, 0.0, 0.32, pitch=2.35619, yaw=0.0, wait=True, duration=2.5)
        time.sleep(0.5)
        
        # 记录伸展位姿下的完整关节角度，然后仅修改 Joint 1 添加旋转
        extended_joints = self.arm.current_joints.copy()
        extended_joints[0] += angle_rad
        
        # 在关节空间中执行旋转（不走笛卡尔直线，不经过奇异区）
        self.arm.set_joints(extended_joints, speed_pct=15, duration=2.5)
        time.sleep(0.5)
        
        # 沿当前朝向前伸 10cm
        T = self.arm.get_current_pose_matrix()
        cx, cy, cz = T[:3, 3]
        dist = np.hypot(cx, cy)
        if dist > 0.01:
            direction = np.array([cx, cy]) / dist
            nx = cx + direction[0] * 0.10
            ny = cy + direction[1] * 0.10
            print(f"[Action] 前伸至 ({nx:.3f}, {ny:.3f}, {cz:.3f})...")
            self.arm.set_pose(nx, ny, cz, pitch=3.14159, wait=True, duration=1.5)
        
        print("[Action] 松开夹爪...")
        self.arm.set_gripper(self.arm.GRIPPER_OPEN, duration=1.5)
        
        print("[Action] 动作结束，回归标准位姿...")
        self.go_standby()
        return {'success': True}

    def visualize_scene(self):
        """提取当前点云并保存为本地文件"""
        print("[Visualize] 正在获取当前场景点云并执行本地保存...")
        success = self.perception.visualize_current_pointcloud()
        if not success:
            print("[Visualize] 错误: 获取点云失败，可能是深度传感器无数据或距离过近被截断。")
            return {'success': False, 'error': 'Pointcloud retrieval failed'}
        return {'success': True, 'action_completed': True}

    def shutdown(self):
        try:
            self.perception.close()
        except Exception:
            pass
        try:
            self.arm.destroy_node()
        except Exception:
            pass
