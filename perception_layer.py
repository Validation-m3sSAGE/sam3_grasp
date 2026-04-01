import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import requests

D405_DEPTH_MIN_MM = 10
D405_DEPTH_MAX_MM = 1000

class PerceptionLayer:
    def __init__(self,
                 camera_width: int = 848,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 # ↓↓↓ 必须在这里填写您的 GPU 主机局域网 IP 地址 ↓↓↓
                 server_ip: str = "127.0.0.1", 
                 server_port: int = 8000):

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, camera_width, camera_height, rs.format.bgr8, camera_fps)
        cfg.enable_stream(rs.stream.depth, camera_width, camera_height, rs.format.z16, camera_fps)
        self.profile = self.pipeline.start(cfg)
        
        self.align = rs.align(rs.stream.color)
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # 挂载远程 GPU 服务端推理地址
        self.server_url = f"http://{server_ip}:{server_port}/predict"
        print(f"[Perception] 边缘端感知层初始化完成，远程算力挂载点: {self.server_url}")

    def get_chessboard_pose(self, pattern_size=(11, 8), square_size=0.02):
        """提取棋盘格在相机坐标系下的 6D 位姿 (默认 20mm 方格)"""
        color_img, _ = self.get_aligned_frames()
        if color_img is None:
            return False, None
            
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # 寻找棋盘格内角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 生成棋盘格三维物理坐标系 (Z=0 平面)
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            objp *= square_size
            
            camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))
            
            success, rvec, tvec = cv2.solvePnP(objp, corners_subpix, camera_matrix, dist_coeffs)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                T_cam2board = np.eye(4)
                T_cam2board[:3, :3] = R
                T_cam2board[:3, 3] = tvec.flatten()
                return True, T_cam2board
        return False, None

    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
            
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        return color_img, depth_img

    def get_scene_pointcloud(self, downsample_voxel: float = 0.01, min_depth_mm: float = 120.0):
        """获取当前场景的降采样稠密点云对象。引入 min_depth 过滤相机近处的机械臂自身干扰"""
        color_img, depth_img = self.get_aligned_frames()
        if color_img is None:
            return False, None
            
        # 根据传感器尺度将物理毫米阈值逆向转换为硬件原始读数单位
        raw_min_depth = min_depth_mm / (self.depth_scale * 1000.0)
        depth_filtered = depth_img.copy()
        depth_filtered[depth_filtered < raw_min_depth] = 0

        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_filtered),
            depth_scale=1.0 / self.depth_scale,
            depth_trunc=D405_DEPTH_MAX_MM / 1000.0,
            convert_rgb_to_intensity=False
        )
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            color_img.shape[1], color_img.shape[0], self.fx, self.fy, self.cx, self.cy
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
        
        # 点云有效性防空拦截：若有效点数过少，直接返回失败，防止 Open3D 底层崩溃
        if len(pcd.points) < 100:
            return False, None
            
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        return True, pcd

    def register_pointclouds(self, pcd_source, pcd_target, distance_threshold=0.15):
        """利用 ICP 算法配准两帧点云，返回相机坐标系的变换矩阵"""
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target, distance_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        
        fitness = reg_p2p.fitness
        if fitness < 0.8:
            return False, None, fitness
            
        return True, reg_p2p.transformation, fitness

    def extract_target_mask(self, color_img: np.ndarray, target_class: str):
        # 边缘端进行高比例有损压缩，降低上行网络带宽负担
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, encoded_img = cv2.imencode('.jpg', color_img, encode_param)
        if not success:
            print("[Perception] 图像 JPEG 压缩失败。")
            return False, None

        files = {'image': ('capture.jpg', encoded_img.tobytes(), 'image/jpeg')}
        data = {'target_class': target_class}
        
        try:
            # 向 GPU 主机发送跨设备请求，设定 10 秒超时
            response = requests.post(self.server_url, files=files, data=data, timeout=10.0)
            
            if response.status_code == 404:
                return False, None
            elif response.status_code != 200:
                print(f"[Perception] 远程推理服务端异常，状态码: {response.status_code}")
                return False, None
                
            # 接收主机回传的无损 PNG 掩码流并解码
            mask_array = np.frombuffer(response.content, np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print("[Perception] 服务端回传掩码解析失败。")
                return False, None
                
            return True, mask
            
        except requests.exceptions.RequestException as e:
            print(f"[Perception] 网络请求崩溃 (主机未启动或 IP 错误): {e}")
            return False, None

    def compute_pointcloud_centroid(self, mask: np.ndarray, depth_img: np.ndarray, color_img: np.ndarray = None, max_points: int = 500):
        # 强制导出掩码区域内的所有原始点云（包含被判定为无效深度的点）以供排查
        mask_v, mask_u = np.where(mask > 0)
        if len(mask_u) > 0:
            raw_z = depth_img[mask_v, mask_u].astype(float) * self.depth_scale
            raw_x = (mask_u - self.cx) * raw_z / self.fx
            raw_y = (mask_v - self.cy) * raw_z / self.fy
            raw_points = np.stack((raw_x, raw_y, raw_z), axis=-1)
            
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_points)
            if color_img is not None:
                colors_bgr = color_img[mask_v, mask_u]
                colors_rgb = colors_bgr[:, ::-1] / 255.0
                
                # 将深度不在合法范围内的无效点强行标记为纯红色
                invalid_mask = (depth_img[mask_v, mask_u] < D405_DEPTH_MIN_MM) | (depth_img[mask_v, mask_u] > D405_DEPTH_MAX_MM)
                colors_rgb[invalid_mask] = [1.0, 0.0, 0.0]
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            
            save_path = "debug_raw_masked_pointcloud.ply"
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"[Perception] [Debug] 掩码覆盖区域的局部点云(无效深度已标红)已保存至: {save_path}")

        # 将原始深度读数转换为物理毫米 (mm)，对齐常量阈值的量纲
        depth_in_mm = depth_img.astype(float) * self.depth_scale * 1000.0
        valid_depth = (depth_in_mm >= D405_DEPTH_MIN_MM) & (depth_in_mm <= D405_DEPTH_MAX_MM)
        target_region = (mask > 0) & valid_depth
        
        v, u = np.where(target_region)
        if len(u) == 0: return False, None

        z = depth_img[v, u].astype(float) * self.depth_scale
        step = max(1, len(u) // max_points)
        u_sparse, v_sparse, z_sparse = u[::step], v[::step], z[::step]

        x = (u_sparse - self.cx) * z_sparse / self.fx
        y = (v_sparse - self.cy) * z_sparse / self.fy
        
        point_cloud = np.stack((x, y, z_sparse), axis=-1)
        centroid = np.mean(point_cloud, axis=0)
        return True, centroid

    def compute_grasp_yaw(self, mask: np.ndarray):
        """基于2D掩码计算外接矩形偏航角，使夹爪垂直于物体长轴"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        _, (w, h), angle = rect
        
        # minAreaRect 的角度约束：若宽小于高，则需旋转 90 度以对齐短轴
        if w < h:
            angle += 90.0
            
        return np.radians(angle)

    def process(self, target_class: str, visualize: bool = False):
        print(f"[Perception] 开始捕获图像，向 SAM3 下发文本指令进行零样本分割: {target_class}")
        color_img, depth_img = self.get_aligned_frames()
        if color_img is None:
            print("[Perception] 错误: 无法获取相机的彩色/深度流数据")
            return {'success': False, 'error': '流异常'}

        found, mask = self.extract_target_mask(color_img, target_class)
        if not found:
            print(f"[Perception] 警告: 未能根据文本 '{target_class}' 提取到实体掩码")
            return {'success': False, 'error': '未分割到目标'}
        print("[Perception] 目标掩码提取成功")

        # 强制落盘调试图：无论后续深度是否有效，先将掩码与原图叠加并保存
        debug_img = color_img.copy()
        if mask is not None:
            colored_mask = np.zeros_like(color_img)
            colored_mask[mask > 0] = [0, 255, 0]  # 使用半透明绿色覆盖掩码区域
            cv2.addWeighted(colored_mask, 0.4, debug_img, 0.6, 0, debug_img)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        debug_save_path = "debug_mask_rgb_overlay.jpg"
        cv2.imwrite(debug_save_path, debug_img)
        print(f"[Perception] [Debug] 掩码叠加实景图已强制保存至: {debug_save_path}")

        # 传入 color_img 以便生成带有真实颜色及错误红色标记的局部点云
        valid_pose, centroid = self.compute_pointcloud_centroid(mask, depth_img, color_img)
        if not valid_pose:
            print("[Perception] 警告: 目标区域内无有效深度，无法计算 3D 重心")
            return {'success': False, 'error': '目标区域无深度'}

        yaw_rad = self.compute_grasp_yaw(mask)
        print(f"[Perception] 提取完成 -> 3D 重心(相机系): {np.round(centroid, 4)} m | Yaw: {np.degrees(yaw_rad):.2f}°")

        result = {
            'success': True, 'grasp_point_cam': centroid,
            'color_image': color_img, 'mask': mask, 'bbox': None,
            'yaw': yaw_rad
        }

        if visualize:
            vis_img = color_img.copy()
            if mask is not None:
                # 创建半透明掩码覆盖层
                colored_mask = np.zeros_like(color_img)
                colored_mask[mask > 0] = [0, 255, 0]  # 使用绿色填充掩码区域
                cv2.addWeighted(colored_mask, 0.4, vis_img, 0.6, 0, vis_img)
                
                # 绘制掩码边缘轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
                
            info = f"Cam: X={centroid[0]:.3f}, Y={centroid[1]:.3f}, Z={centroid[2]:.3f}m, Yaw={np.degrees(yaw_rad):.1f}deg"
            cv2.putText(vis_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 保存图像到当前路径
            save_path = "mask_visualization.jpg"
            cv2.imwrite(save_path, vis_img)
            print(f"[Perception] 可视化结果已保存至当前路径: {save_path}")
            
            cv2.imshow('Perception View', vis_img)
            cv2.waitKey(1)
            
        return result

    def visualize_current_pointcloud(self):
        """获取当前帧点云并保存到本地（支持无头环境）"""
        success, pcd = self.get_scene_pointcloud(downsample_voxel=0.005)
        if success and pcd is not None:
            save_path = "current_scene_pointcloud.ply"
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"[Perception] 点云数据已保存至当前目录: {save_path}")
            return True
        return False

    def close(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
