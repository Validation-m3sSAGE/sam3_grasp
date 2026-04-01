import open3d as o3d
import os
import sys

def main():
    file_path = "debug_raw_masked_pointcloud.ply"
    
    if not os.path.exists(file_path):
        print(f"[Error] 点云文件未找到: {file_path}")
        print("请确保主程序已成功执行并保存了点云数据。")
        sys.exit(1)
        
    print(f"正在读取点云文件: {file_path} ...")
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"[Error] 文件读取异常: {e}")
        sys.exit(1)
    
    if pcd.is_empty():
        print("[Error] 读取失败，点云数据为空或文件已损坏。")
        sys.exit(1)
        
    print(f"读取成功。当前点云包含点数: {len(pcd.points)}")
    print("正在启动 Open3D 可视化窗口 (鼠标拖拽旋转，滚轮缩放，按 'Q' 或 'Esc' 退出)...")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Saved PointCloud Viewer",
        width=1024,
        height=768,
        point_show_normal=False
    )

if __name__ == "__main__":
    main()
