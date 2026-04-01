import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import torch
# 引入 Meta 官方 SAM3 原生运行库
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

D405_DEPTH_MIN_MM = 70
D405_DEPTH_MAX_MM = 2000

class PerceptionLayer:
    def __init__(self,
                 camera_width: int = 848,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 sam_model: str = 'sam3.pt'):

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

        # 初始化 Meta 官方原生模型与推理处理器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam3_model = build_sam3_image_model(checkpoint_path=sam_model, device=device)
        self.sam3_processor = Sam3Processor(self.sam3_model, device=device)

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
        from PIL import Image
        # 转换颜色空间以符合内部预处理逻辑
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # 将 NumPy 的 HWC 格式强制转换为 PIL Image
        # 规避官方 Sam3Processor 对 ndarray shape[-2:] 的宽、通道错误解析
        pil_img = Image.fromarray(color_rgb)
        
        # 依赖 Meta 原生库的处理器进行图像设置与文本提示
        state = self.sam3_processor.set_image(pil_img)
        state = self.sam3_processor.set_text_prompt(target_class, state)
        
        if "masks" not in state or len(state["masks"]) == 0:
            return False, None

        masks = state["masks"]
        scores = state["scores"]
        
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
        
        # 展平置信度，提取最佳掩码索引
        scores_np = scores_np.flatten()
        best_idx = int(np.argmax(scores_np))
        
        if best_idx >= masks_np.shape[0]:
            best_idx = 0
            
        # 传入 PIL 图像后，底层将稳定输出无缩放畸变的 (N, 1, 480, 848) 或 (N, 480, 848) 掩码张量
        # 去除无效的兜底重整，直接降维剥离
        best_mask = np.squeeze(masks_np[best_idx])
        
        # 提取掩码并格式化为 8 位图像
        mask = (best_mask > 0).astype(np.uint8) * 255
        
        return True, mask

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
