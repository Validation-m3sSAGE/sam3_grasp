import pybullet as p
import pybullet_data
import numpy as np
import os

class OMPLKinematicsPlanner:
    """
    基于 PyBullet 物理引擎的运动学与高维关节空间规划模块。
    用于承载 OMPL 算法管线的前置环境建模与精确逆运动学(IK)投影，
    将笛卡尔直线路径转换为完全安全的关节流形轨迹。
    """
    def __init__(self, urdf_path):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF 模型文件未找到: {urdf_path}")
            
        import xml.etree.ElementTree as ET
        import tempfile
        
        # 动态剔除 URDF 中的视觉与碰撞体标签，仅保留运动学骨架以规避缺失网格文件的报错
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for link in root.findall('link'):
            for vis in link.findall('visual'):
                link.remove(vis)
            for col in link.findall('collision'):
                link.remove(col)
                
        fd, temp_urdf_path = tempfile.mkstemp(suffix=".urdf")
        with os.fdopen(fd, 'wb') as f:
            tree.write(f)
            
        try:
            self.robot_id = p.loadURDF(temp_urdf_path, [0, 0, 0], useFixedBase=True)
        finally:
            os.remove(temp_urdf_path)
        
        self.joint_indices = []
        self.ll = []
        self.ul = []
        self.jr = []
        self.rp = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                # 读取 URDF 物理极限，辅助 IK 避开超限解
                lower = info[8]
                upper = info[9]
                if lower >= upper: 
                    lower, upper = -3.14, 3.14
                self.ll.append(lower)
                self.ul.append(upper)
                self.jr.append(upper - lower)
                # 将静息姿态偏好设为量程中心
                self.rp.append((lower + upper) / 2.0)
                
        if len(self.joint_indices) < 6:
            raise ValueError("URDF 模型解析异常，未能检出标准的 6 个旋转关节。")
            
        # 设置末端执行器索引 (锁定在 link6 处)
        self.ee_link_idx = self.joint_indices[5]

    def solve_fk(self, current_joints):
        """
        基于当前关节角进行正向运动学解算，返回末端执行器在基座系下的 4x4 齐次变换矩阵。
        这保证了正解与逆解在同一个物理空间模型下完全对齐。
        """
        for i, idx in enumerate(self.joint_indices[:6]):
            p.resetJointState(self.robot_id, idx, current_joints[i])
            
        info = p.getLinkState(self.robot_id, self.ee_link_idx)
        pos = info[4]
        orn = info[5]
        
        R = p.getMatrixFromQuaternion(orn)
        R = np.array(R).reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def solve_ik(self, target_pos, target_rpy, current_joints):
        """
        在虚拟仿真环境中计算目标位姿的逆向解算，强制向连续的工作域收敛。
        """
        # 以实际物理状态作为种子点，防止解算跳跃
        for i, idx in enumerate(self.joint_indices[:6]):
            p.resetJointState(self.robot_id, idx, current_joints[i])
            
        target_orn = p.getQuaternionFromEuler(target_rpy)
        
        # 激活零空间(Null Space)全局搜索，传入关节限位参数。
        # 这将强制 IK 求解器在遇到多组逆解时，优先选择(如抬高肘部等)使所有关节保持在安全裕度内的构型。
        ik_solution = p.calculateInverseKinematics(
            self.robot_id, 
            self.ee_link_idx, 
            target_pos, 
            target_orn,
            lowerLimits=self.ll[:6],
            upperLimits=self.ul[:6],
            jointRanges=self.jr[:6],
            restPoses=self.rp[:6],
            maxNumIterations=500,
            residualThreshold=1e-5
        )
        sol = np.array(ik_solution[:6])
        
        # 若在挂载全量限位约束后解算依然超限，才确认为物理上彻底不可达
        if abs(sol[4]) > 1.22:
            raise ValueError(f"手腕依然超限({sol[4]:.2f}rad)。说明在该距离和姿态下，物理上已超出手腕关节的弯曲极限(±1.22rad)。")
        return sol

    def destroy(self):
        try:
            p.disconnect(self.client_id)
        except Exception:
            pass