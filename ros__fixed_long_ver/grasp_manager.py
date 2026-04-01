import numpy as np
import time
import threading
from rclpy.executors import MultiThreadedExecutor

from perception_layer import PerceptionLayer
from arm_controller import PiperArmController

# 机械臂夹爪中心在相机坐标系下的近似偏置 (X, Y, Z 单位: 米)
# 相机坐标系常规约定：+X为右方，+Y为下方，+Z为前方。
# X为负值表示夹爪偏向相机的左侧。可根据实际物理组装距离进行微调。
GRIPPER_OFFSET_IN_CAM = np.array([-0.03, 0.05, 0.10])

# --- 视觉伺服核心参数（方便手动微调） ---
SERVO_SAFE_Z = 0.20       # 观测安全高度，保持在目标上方 20cm，规避近距离盲区
SERVO_TOLERANCE = 0.005   # XY 平面对准容差，缩小至 5mm 强制增加逼近迭代次数
SERVO_KP = 0.5            # 比例控制系数

import os

class GraspManager:
    def __init__(self, calib_path=None):
        self.executor = MultiThreadedExecutor()
        self.arm = PiperArmController()
        self.executor.add_node(self.arm)
        self._ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._ros_thread.start()

        time.sleep(0.5)
        self.perception = PerceptionLayer()
        self.calib_path = calib_path or "calibration_result.npz"
        self.R_cam2base = None
        
        # 启动时默认尝试加载历史标定文件
        if os.path.exists(self.calib_path):
            try:
                data = np.load(self.calib_path)
                self.R_cam2base = data['R_cam2base']
                print(f"[Init] 成功加载历史手眼标定矩阵: {self.calib_path}")
            except Exception as e:
                print(f"[Init] 历史标定文件读取失败 ({e})，请稍后手动执行标定。")

    def _spin_ros(self):
        try:
            self.executor.spin()
        except Exception:
            pass

    def calibrate_axes(self, move_dist=0.05):
        """双向多点标定，基于点云 ICP 配准计算相机系到基座系的坐标转换矩阵"""
        print("[Calibration] 开始高精度双向全轴标定，请确保视野内放置几何特征丰富的测试物体...")
        self.arm.enable_arm(True)
        
        if not self.arm.wait_for_pose(timeout=5.0):
            raise RuntimeError("未收到机械臂末端位姿反馈")

        # 将初始位置微调至工作空间中心，防止双向运动时发生负向越界
        print("[Calibration] 移动至初始观测位姿...")
        init_x, init_y, init_z = 0.20, 0.0, 0.25
        self.arm.set_pose(init_x, init_y, init_z, wait=True, duration=3.0)
        time.sleep(1.0)

        C_matrix = np.zeros((3, 3))
        axes = ['x', 'y', 'z']
        
        for i, axis in enumerate(axes):
            print(f"[Calibration] 标定基座 {axis.upper()} 轴...")
            
            # 针对 Z 轴：为防止向上抬升导致目标丢失视野，将 Z 轴的标定基准面整体下沉
            current_z = init_z - move_dist if axis == 'z' else init_z
            
            # 1. 采集基准点云 (提高分辨率至5mm，增加静息等待时间消除机械余震)
            self.arm.set_pose(init_x, init_y, current_z, wait=True, duration=2.0)
            time.sleep(1.5)
            success_center, pcd_center = self.perception.get_scene_pointcloud(downsample_voxel=0.005)
            if not success_center:
                raise RuntimeError(f"标定 {axis} 轴：基准点云获取失败")

            # 2. 正向微动并采点
            target_pos_plus = [init_x, init_y, current_z]
            target_pos_plus[i] += move_dist
            self.arm.set_pose(*target_pos_plus, wait=True, duration=1.5)
            time.sleep(1.5)
            success_plus, pcd_plus = self.perception.get_scene_pointcloud(downsample_voxel=0.005)
            if not success_plus:
                raise RuntimeError(f"标定 {axis} 轴：正向点云获取失败")
                
            success_reg_plus, T_plus, fit_plus = self.perception.register_pointclouds(pcd_center, pcd_plus)
            if not success_reg_plus:
                raise RuntimeError(f"标定 {axis} 轴：正向配准精度不足 (Fitness: {fit_plus:.4f})")
            t_plus = T_plus[:3, 3]

            # 3. 负向微动并采点
            target_pos_minus = [init_x, init_y, current_z]
            target_pos_minus[i] -= move_dist
            self.arm.set_pose(*target_pos_minus, wait=True, duration=2.0)
            time.sleep(1.5)
            success_minus, pcd_minus = self.perception.get_scene_pointcloud(downsample_voxel=0.005)
            if not success_minus:
                raise RuntimeError(f"标定 {axis} 轴：负向点云获取失败")
                
            success_reg_minus, T_minus, fit_minus = self.perception.register_pointclouds(pcd_center, pcd_minus)
            if not success_reg_minus:
                raise RuntimeError(f"标定 {axis} 轴：负向配准精度不足 (Fitness: {fit_minus:.4f})")
            t_minus = T_minus[:3, 3]

            # 4. 双向差分计算观测向量
            # 物理总位移为 2 * move_dist，方向向量为 -(t_plus - t_minus)
            c_axis = -(t_plus - t_minus) / (2 * move_dist)
            C_matrix[:, i] = c_axis
            
            print(f"[Calibration] {axis.upper()} 轴双向配准完成 (Fit+: {fit_plus:.4f}, Fit-: {fit_minus:.4f})")
            print(f"[Calibration] 观测位移向量: {np.round(c_axis, 4)}")

        # 恢复至观测基准位姿
        self.arm.set_pose(init_x, init_y, init_z, wait=True, duration=1.5)

        try:
            # 对配准生成的标定矩阵执行 SVD 正交化，消除缩放畸变，将其投影至纯旋转空间 SO(3)
            U, _, Vt = np.linalg.svd(C_matrix)
            C_orthogonal = U @ Vt
            
            # 若行列式小于 0，说明产生了镜像反射，需对特征向量进行物理空间翻转
            if np.linalg.det(C_orthogonal) < 0:
                U[:, 2] *= -1
                C_orthogonal = U @ Vt
                
            # 矩阵求逆（正交矩阵的逆等于其转置），得到纯旋转的坐标映射矩阵
            self.R_cam2base = np.linalg.inv(C_orthogonal)
        except np.linalg.LinAlgError:
            raise RuntimeError("标定矩阵奇异，相机与机械臂坐标映射求解失败")

        print("[Calibration] 全轴标定完成！\n相机系 -> 基座系转换矩阵 (R_cam2base):")
        print(np.round(self.R_cam2base, 4))
        
        # 校验矩阵的三轴尺度因子，正常物理映射的模长应当接近 1.0
        scale_x = np.linalg.norm(self.R_cam2base[:, 0])
        scale_y = np.linalg.norm(self.R_cam2base[:, 1])
        scale_z = np.linalg.norm(self.R_cam2base[:, 2])
        print(f"[Calibration] 映射矩阵尺度因子: X={scale_x:.2f}, Y={scale_y:.2f}, Z={scale_z:.2f}")
        if any(s > 2.0 or s < 0.5 for s in [scale_x, scale_y, scale_z]):
            print("[Calibration] 警告：标定矩阵存在尺度畸变，可能由于点云 ICP 配准不佳导致后续坐标运算被异常放大。")

        # 标定完成后覆盖保存，供后续启动默认加载
        np.savez(self.calib_path, R_cam2base=self.R_cam2base)
        print(f"[Calibration] 矩阵已成功持久化至: {self.calib_path}")

        return self.R_cam2base

    def grasp_simple(self, target_name='apple'):
        """单次直接抓取执行管线"""
        print(f"[Grasp Simple] 开始执行单次抓取，目标: {target_name}")
        self.arm.enable_arm(True)
        self.arm.set_gripper(self.arm.GRIPPER_OPEN)

        if self.R_cam2base is None:
            print("[Error] 缺失坐标转换矩阵且未加载到历史文件，请先执行手动标定。")
            return {'success': False, 'error': 'Missing calibration matrix'}

        result = self.perception.process(target_name, visualize=True)
        if not result['success']:
            error_msg = result.get('error', '未知异常')
            print(f"[Grasp Simple] 错误: 感知管线执行失败 ({error_msg})")
            raise RuntimeError(f"感知失败: {error_msg}")

        p_target_cam = result['grasp_point_cam']
        print(f"[Grasp Simple] 目标 3D 坐标 (相机系): {np.round(p_target_cam, 4)} m")

        delta_cam = p_target_cam - GRIPPER_OFFSET_IN_CAM
        print(f"[Grasp Simple] 减去夹爪偏置 {GRIPPER_OFFSET_IN_CAM} 后，相机系偏差: {np.round(delta_cam, 4)} m")
        
        delta_base = self.R_cam2base @ delta_cam
        print(f"[Grasp Simple] 转换至基座系，运动增量: {np.round(delta_base, 4)} m")

        T_curr = self.arm.get_current_pose_matrix()
        curr_x, curr_y, curr_z = T_curr[:3, 3]
        
        target_x = curr_x + delta_base[0]
        target_y = curr_y + delta_base[1]
        target_z = curr_z + delta_base[2]
        
        # 根据当前夹爪坐标与映射矩阵，反推相机光心在基座系下的绝对坐标
        cam_base_offset = self.R_cam2base @ GRIPPER_OFFSET_IN_CAM
        cam_x = curr_x - cam_base_offset[0]
        cam_y = curr_y - cam_base_offset[1]
        cam_z = curr_z - cam_base_offset[2]
        
        print(f"[Grasp Simple] 当前夹爪末端位姿 (基座系): X={curr_x:.4f}, Y={curr_y:.4f}, Z={curr_z:.4f}")
        print(f"[Grasp Simple] 预估相机光心坐标 (基座系): X={cam_x:.4f}, Y={cam_y:.4f}, Z={cam_z:.4f}")
        print(f"[Grasp Simple] 目标绝对物理坐标 (基座系): X={target_x:.4f}, Y={target_y:.4f}, Z={target_z:.4f}")

        # 物理工作空间强制安全拦截 (放宽最大平距限制至 0.45m)
        target_radius = np.hypot(target_x, target_y)
        if target_radius > 1. or target_z < -1. or target_z > 1.:
            print(f"[Error] 目标点超限！距基座平距 {target_radius:.3f}m, 高度 {target_z:.3f}m。超出物理工作空间，拒绝执行抓取。")
            return {'success': False, 'error': 'Target out of physical workspace boundaries'}

        print("[Action] 移动至目标上方预抓取点 (Z + 0.05m)...")
        success = self.arm.set_pose(target_x, target_y, target_z + 0.05, wait=True, duration=2.5)
        if not success: raise RuntimeError("机械臂上方位姿移动指令执行异常")

        print("[Action] 下放至夹取高度...")
        success = self.arm.set_pose(target_x, target_y, target_z, wait=True, duration=1.5)
        if not success: raise RuntimeError("机械臂下放指令执行异常")

        print("[Action] 闭合夹爪...")
        self.arm.set_gripper(self.arm.GRIPPER_CLOSE, duration=1.5)

        print("[Action] 抬起机械臂复位 (Z + 0.15m)...")
        self.arm.set_pose(target_x, target_y, target_z + 0.15, wait=True, duration=2.0)

        print("[Grasp Simple] 抓取动作链执行完毕。")
        return {'success': True, 'action_completed': True}

    def grasp_servo(self, target_name='apple', max_iter=20):
        """分段式视觉伺服抓取：先在安全高度完成平面迭代对准，最后开环盲抓"""
        self.arm.enable_arm(True)
        self.arm.set_gripper(self.arm.GRIPPER_OPEN)

        if self.R_cam2base is None:
            print("[Error] 缺失坐标转换矩阵且未加载到历史文件，请先执行手动标定。")
            return {'success': False, 'error': 'Missing calibration matrix'}

        print("[Servo] 开始视觉伺服迭代对准...")
        last_target_base = None
        last_yaw_target = 0.0

        for i in range(max_iter):
            result = self.perception.process(target_name, visualize=True)
            if not result['success']:
                print(f"[Servo] 迭代 {i+1}: 感知异常 - {result.get('error')}")
                if last_target_base is not None:
                    print("[Servo] 目标脱离视野中心或深度失效，利用历史 3D 坐标强制进入盲抓。")
                    break
                else:
                    return {'success': False, 'error': '初始观测失败，无法建立目标位置'}
            
            p_target_cam = result['grasp_point_cam']
            yaw_target = result.get('yaw', 0.0)
            
            # 计算相机系下偏差
            delta_cam = p_target_cam - GRIPPER_OFFSET_IN_CAM
            error_dist = np.hypot(delta_cam[0], delta_cam[1])
            
            print(f"[Servo] 迭代 {i+1}: XY 偏差={error_dist:.4f} m (容差:{SERVO_TOLERANCE} m), 观测目标 Z={p_target_cam[2]:.4f} m")

            # 转换为基座坐标系增量
            delta_base = self.R_cam2base @ delta_cam
            print(f"[Servo] 迭代 {i+1}: 基座系运动补偿增量 [dx, dy, dz] = {np.round(delta_base, 4)}")
            
            T_curr = self.arm.get_current_pose_matrix()
            curr_x, curr_y, curr_z = T_curr[:3, 3]

            # 记录目标的物理绝对坐标 (X, Y, Z) 与 Yaw
            last_target_base = np.array([
                curr_x + delta_base[0],
                curr_y + delta_base[1],
                curr_z + delta_base[2]
            ])
            last_yaw_target = yaw_target
            print(f"[Servo] 迭代 {i+1}: 更新目标全局锚点为 X={last_target_base[0]:.4f}, Y={last_target_base[1]:.4f}, Z={last_target_base[2]:.4f}")

            # 判定收敛，若偏差足够小则退出伺服
            if error_dist < SERVO_TOLERANCE:
                print("[Servo] 对准完成，当前偏差已满足控制要求。")
                break
                
            # P 控制：水平面逼近，高度维持安全截断距离，保持垂直姿态
            move_x = curr_x + delta_base[0] * SERVO_KP
            move_y = curr_y + delta_base[1] * SERVO_KP
            target_z_safe = last_target_base[2] + SERVO_SAFE_Z
            
            # 工作空间安全检查
            move_radius = np.hypot(move_x, move_y)
            if move_radius > 0.80:
                raise RuntimeError(f"伺服坐标发散：水平补偿半径达 {move_radius:.4f} m，超出物理工作极限 (0.80m)。")
            
            self.arm.set_pose(move_x, move_y, target_z_safe, wait=True, duration=1.5)
            time.sleep(0.5)
        else:
            print("[Servo] 警告: 达到最大迭代次数未完全收敛，执行强制盲抓。")

        if last_target_base is None:
            raise RuntimeError("伺服期间未建立有效的空间坐标，无法执行盲抓。")

        # --- 盲抓阶段 ---
        print(f"[Blind Grasp] 退出伺服阶段。提取历史锚点 (基座系): X={last_target_base[0]:.4f}, Y={last_target_base[1]:.4f}, Z={last_target_base[2]:.4f}")
        target_x, target_y, target_z = last_target_base

        print(f"[Action] 垂直下放至 Z={target_z:.4f} m，对齐短轴 Yaw={last_yaw_target:.2f} rad...")
        success = self.arm.set_pose(target_x, target_y, target_z, yaw=last_yaw_target, wait=True, duration=2.5)
        if not success:
            raise RuntimeError("盲抓下放动作执行异常")

        print("[Action] 闭合夹爪...")
        self.arm.set_gripper(self.arm.GRIPPER_CLOSE, duration=1.5)

        print("[Action] 抬起机械臂复位...")
        self.arm.set_pose(target_x, target_y, target_z + 0.20, yaw=0.0, wait=True, duration=2.0)

        return {'success': True, 'action_completed': True}

    def estimate_pose(self, target_name='apple'):
        return {'success': False, 'error': '该模式尚未实现'}

    def search_and_grasp(self, target_name='apple'):
        return {'success': False, 'error': '该模式尚未实现'}

    def visualize_scene(self):
        """提取并阻塞渲染当前点云，用以审查场景特征"""
        print("[Visualize] 正在获取并渲染当前场景点云，请在弹出的窗口中审查三维特征（关闭窗口以继续）...")
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
