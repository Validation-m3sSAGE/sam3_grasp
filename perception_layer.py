import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import requests

D405_DEPTH_MIN_MM = 70
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
            
        # 降低过滤阈值至 120mm，防止将高出桌面的目标物体（如苹果顶部）错误滤除
        depth_filtered = depth_img.copy()
        depth_filtered[depth_filtered < min_depth_mm] = 0

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

    def compute_pointcloud_centroid(self, mask: np.ndarray, depth_img: np.ndarray, max_points: int = 500):
        valid_depth = (depth_img >= D405_DEPTH_MIN_MM) & (depth_img <= D405_DEPTH_MAX_MM)
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

        valid_pose, centroid = self.compute_pointcloud_centroid(mask, depth_img)
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
        """获取当前帧点云并调用 Open3D 窗口渲染"""
        success, pcd = self.get_scene_pointcloud(downsample_voxel=0.005)
        if success and pcd is not None:
            o3d.visualization.draw_geometries([pcd], window_name="Current Scene PointCloud")
            return True
        return False

    def close(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
