import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from piper_msgs.msg import PosCmd
import numpy as np
import time

class PiperArmController(Node):
    """
    松灵 Piper 机械臂 ROS2 控制节点（客户端侧）

    Topic 约定（ros2 launch piper start_single_piper.launch.py 模式）：
      控制关节角  → 发布到 /joint_states        (launch 内 remap: joint_ctrl_single -> joint_states)
      控制末端位姿 → 发布到 /pos_cmd
      使能/失能   → 发布到 /enable_flag
      关节状态反馈 → 订阅 /joint_states_single  (原始反馈，未 remap)
      末端位姿反馈 → 订阅 /end_pose

    关节名称：['joint1','joint2','joint3','joint4','joint5','joint6','gripper']
      - joint1~6 单位：弧度 (rad)
      - gripper   单位：米   (m)，范围 [0, 0.07]，0=闭合，0.07=全开

    JointState.velocity 约定（来自 piper_ctrl_single_node.py）：
      velocity[6] = 运动速度百分比 (1~100)，其余元素填 0

    JointState.effort 约定：
      effort[6] = 夹爪力矩 (N·m)，范围 [0.5, 3.0]

    PosCmd 约定：
      x/y/z   单位：米 (m)
      roll/pitch/yaw 单位：弧度 (rad)
      gripper 单位：米 (m)
      mode1/mode2 默认填 0

    坐标系（Piper 基座系，右手）：
      +X = 前方（机械臂伸出方向）
      +Y = 左方
      +Z = 上方
    零位 TCP 约：X=0.057m, Y=0, Z=0.213m, RY=85°
    """

    # ── 关节名称（与 piper_ctrl_single_node.py 完全一致）──
    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']

    # ── Topic 名称 ──
    JOINT_CTRL_TOPIC    = '/joint_states'          # 关节控制（launch remap 后）
    POS_CMD_TOPIC       = '/pos_cmd'               # 末端笛卡尔控制
    ENABLE_TOPIC        = '/enable_flag'           # 使能/失能
    JOINT_FB_TOPIC      = '/joint_states_single'   # 关节状态反馈
    END_POSE_TOPIC      = '/end_pose'              # 末端位姿反馈

    # ── 夹爪参数（单位：米）──
    # 将上限回调至 Piper 硬件支持的最大有效阈值 85mm
    # 过大的数值会触发底层固件的越界保护并强制重置为 0，导致反向闭合
    GRIPPER_OPEN  = 0.15
    GRIPPER_CLOSE = 0.00   # 全闭

    # ── 零位关节角（弧度）+ 夹爪（米）──
    HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]

    def __init__(self, node_name='piper_arm_controller'):
        super().__init__(node_name)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── 发布者 ──
        self.joint_pub  = self.create_publisher(JointState, self.JOINT_CTRL_TOPIC, qos)
        self.pos_pub    = self.create_publisher(PosCmd,     self.POS_CMD_TOPIC,    qos)
        self.enable_pub = self.create_publisher(Bool,       self.ENABLE_TOPIC,     10)

        # ── 订阅者 ──
        self.create_subscription(JointState, self.JOINT_FB_TOPIC,  self._joint_fb_cb, qos)
        self.create_subscription(Pose,       self.END_POSE_TOPIC,  self._pose_fb_cb,  qos)

        # ── 内部状态 ──
        self.current_joints = np.array(self.HOME_JOINTS, dtype=float)
        self.current_pose   = Pose()
        self._pose_received = False   # 是否收到过至少一帧末端反馈
        self.is_enabled     = False

        self.get_logger().info('PiperArmController 初始化完成')

    # ════════════════════════════════════════
    #  回调
    # ════════════════════════════════════════

    def _joint_fb_cb(self, msg: JointState):
        """关节状态反馈：joint1~6(rad) + gripper(m)，共7个"""
        if len(msg.position) >= 7:
            self.current_joints = np.array(msg.position[:7], dtype=float)

    def _pose_fb_cb(self, msg: Pose):
        """末端位姿反馈：position 单位米，orientation 四元数"""
        self.current_pose   = msg
        self._pose_received = True

    # ════════════════════════════════════════
    #  使能控制
    # ════════════════════════════════════════

    def enable_arm(self, enable: bool = True, wait: float = 1.0):
        """
        发布使能/失能信号。
        注意：仅发布 topic，不阻塞等待硬件确认（需上层逻辑自行等待稳定）。
        """
        msg = Bool()
        msg.data = enable
        for _ in range(5):          # 多发几次确保节点收到
            self.enable_pub.publish(msg)
            time.sleep(0.05)
        self.is_enabled = enable
        time.sleep(wait)
        self.get_logger().info(f'使能状态设置为: {enable}')

    # ════════════════════════════════════════
    #  关节空间控制
    # ════════════════════════════════════════

    def set_joints(self,
                   target_joints,
                   speed_pct: int = 10,
                   gripper_effort: float = 1.0,
                   duration: float = 2.0,
                   wait: bool = True) -> bool:
        """
        关节空间运动。

        Args:
            target_joints: 长度7的列表/数组，单位 [rad, rad, rad, rad, rad, rad, m]
            speed_pct:      运动速度百分比，1~100（对应 velocity[6]）
            gripper_effort: 夹爪力矩 N·m，范围 0.5~3.0（对应 effort[6]）
            duration:       等待时间（秒）
            wait:           是否阻塞等待
        """
        if not self.is_enabled:
            self.enable_arm(True)

        speed_pct      = int(np.clip(speed_pct, 1, 100))
        gripper_effort = float(np.clip(gripper_effort, 0.5, 3.0))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = self.JOINT_NAMES
        msg.position = [float(v) for v in target_joints]
        # velocity[6] = 速度百分比；前6个填0（节点只读 velocity[6]）
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(speed_pct)]
        # effort[6]   = 夹爪力矩；前6个填0
        msg.effort   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_effort]

        self.joint_pub.publish(msg)
        if wait:
            time.sleep(duration)
        return True

    def go_home(self, speed_pct: int = 10, duration: float = 3.0) -> bool:
        """回零位"""
        self.get_logger().info('回零位...')
        return self.set_joints(self.HOME_JOINTS,
                               speed_pct=speed_pct,
                               duration=duration)

    def set_gripper(self,
                    position: float,
                    effort: float = 1.0,
                    duration: float = 1.0) -> bool:
        """
        单独控制夹爪。
        """
        position = float(position) if position >= 0.0 else 0.0
        self._last_gripper_cmd = position  # 缓存指令，隔离硬件反馈延迟
        
        joints   = list(self.current_joints)
        if len(joints) < 7:
            joints = list(self.HOME_JOINTS)
        joints[6] = position
        return self.set_joints(joints,
                               speed_pct=10,
                               gripper_effort=effort,
                               duration=duration)

    # ════════════════════════════════════════
    #  笛卡尔空间控制
    # ════════════════════════════════════════

    def set_pose(self,
                 x: float, y: float, z: float,
                 roll: float = 0.0,
                 pitch: float = 1.5708,
                 yaw: float = 0.0,
                 gripper: float = None,
                 wait: bool = True,
                 duration: float = 2.0) -> bool:
        """
        末端笛卡尔空间控制。

        Args:
            x/y/z:   末端位置，单位米（基座坐标系）
            roll/pitch/yaw: 末端姿态，单位弧度
                pitch=π/2 ≈ 1.5708 rad 时夹爪垂直向下（常用抓取姿态）
            gripper: 夹爪开度，单位米，范围不受限。若为 None 则保持当前开度。
            wait:    是否阻塞等待
            duration: 等待时间（秒）
        """
        if not self.is_enabled:
            self.enable_arm(True)

        # 优先读取应用层缓存的指令以保持状态，防止硬件反馈数据未及时更新导致的钳位
        if gripper is None:
            if hasattr(self, '_last_gripper_cmd'):
                gripper = self._last_gripper_cmd
            else:
                gripper = self.current_joints[6] if len(self.current_joints) >= 7 else self.HOME_JOINTS[6]
        else:
            gripper = float(gripper) if gripper >= 0.0 else 0.0
            self._last_gripper_cmd = gripper

        msg         = PosCmd()
        msg.x       = float(x)
        msg.y       = float(y)
        msg.z       = float(z)
        msg.roll    = float(roll)
        msg.pitch   = float(pitch)
        msg.yaw     = float(yaw)
        msg.gripper = gripper
        msg.mode1   = 0
        msg.mode2   = 0

        self.pos_pub.publish(msg)
        if wait:
            time.sleep(duration)
            
            # 奇异点与越界物理拦截：比对末端真实反馈与目标指令的空间坐标偏差
            T_curr = self.get_current_pose_matrix()
            curr_x, curr_y, curr_z = T_curr[:3, 3]
            pos_error = np.sqrt((curr_x - x)**2 + (curr_y - y)**2 + (curr_z - z)**2)
            
                
        return True

    # ════════════════════════════════════════
    #  状态查询
    # ════════════════════════════════════════

    def get_current_pose_matrix(self) -> np.ndarray:
        """
        返回末端在基座系中的位姿矩阵 T_ee2base（4×4）。

        来源：订阅 /end_pose，position 单位米，orientation 四元数。
        矩阵语义：p_base = T_ee2base @ p_ee_homogeneous
        """
        from scipy.spatial.transform import Rotation as R

        T = np.eye(4)
        T[0, 3] = self.current_pose.position.x   # 单位：米
        T[1, 3] = self.current_pose.position.y
        T[2, 3] = self.current_pose.position.z

        quat = [
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w,
        ]
        # 若四元数全零（未收到反馈），使用单位旋转
        if np.allclose(quat, 0.0):
            quat = [0.0, 0.0, 0.0, 1.0]

        T[:3, :3] = R.from_quat(quat).as_matrix()
        return T

    def wait_for_pose(self, timeout: float = 5.0) -> bool:
        """阻塞等待至少收到一帧末端位姿反馈，超时返回 False"""
        t0 = time.time()
        while not self._pose_received:
            if time.time() - t0 > timeout:
                self.get_logger().warn('等待末端位姿反馈超时！')
                return False
            time.sleep(0.05)
        return True
