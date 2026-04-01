#!/usr/bin/env python3
"""
松灵 Piper + SAM3 抓取系统 入口

使用方式：
  # 交互模式
  python main.py

  # Agent / 命令行直接调用
  python main.py --mode grasp_simple   --target apple
  python main.py --mode grasp_servo    --target apple --max_iter 20
  python main.py --mode estimate_pose  --target apple
  python main.py --mode search_and_grasp --target apple
  python main.py --mode go_home

启动前提：
  终端1: ros2 launch piper start_single_piper.launch.py
  终端2: conda activate sam3_grasp && python main.py
"""

import argparse
import sys
import traceback
import rclpy
from grasp_manager import GraspManager


def interactive_mode(manager):
    while True:
        print()
        print("=" * 50)
        print("  松灵Piper + SAM3 抓取系统")
        print("=" * 50)
        print("  1. 单次直接抓取   (grasp_simple)")
        print("  2. 视觉伺服抓取   (grasp_servo)")
        print("  3. 预估抓取位姿   (estimate_pose)")
        print("  4. 旋转搜索抓取   (search_and_grasp)")
        print("  5. 可视化当前点云 (visualize_scene)")
        print("  6. 执行手眼标定   (calibrate)")
        print("  h. 机械臂回零")
        print("  q. 退出")
        print("-" * 50)

        choice = input("输入选项: ").strip().lower()

        if choice == "1":
            target = input("目标名称 (默认 apple): ").strip() or "apple"
            result = manager.grasp_simple(target)
            print(f"结果: {result}")

        elif choice == "2":
            target   = input("目标名称 (默认 apple): ").strip() or "apple"
            max_iter = input("最大迭代次数 (默认 20): ").strip()
            max_iter = int(max_iter) if max_iter.isdigit() else 20
            result   = manager.grasp_servo(target, max_iter=max_iter)
            print(f"结果: {result}")

        elif choice == "3":
            target = input("目标名称 (默认 apple): ").strip() or "apple"
            result = manager.estimate_pose(target)
            print(f"结果: {result}")

        elif choice == "4":
            target = input("目标名称 (默认 apple): ").strip() or "apple"
            result = manager.search_and_grasp(target)
            print(f"结果: {result}")

        elif choice == "5":
            result = manager.visualize_scene()
            print(f"结果: {result}")

        elif choice == "6":
            try:
                manager.calibrate_axes()
                print("结果: {'success': True}")
            except Exception as e:
                print(f"结果: {{'success': False, 'error': '{str(e)}'}}")

        elif choice == "h":
            manager.arm.go_home()
            print("已回零")

        elif choice == "q":
            break

        else:
            print("无效选项，请重新输入")


def cli_mode(args, manager):
    mode   = args.mode
    target = args.target

    if mode == "grasp_simple":
        result = manager.grasp_simple(target)
    elif mode == "grasp_servo":
        result = manager.grasp_servo(target, max_iter=args.max_iter)
    elif mode == "estimate_pose":
        result = manager.estimate_pose(target)
    elif mode == "search_and_grasp":
        result = manager.search_and_grasp(target)
    elif mode == "visualize_scene":
        result = manager.visualize_scene()
    elif mode == "calibrate":
        try:
            manager.calibrate_axes()
            result = {"success": True}
        except Exception as e:
            result = {"success": False, "error": str(e)}
    elif mode == "go_home":
        manager.arm.go_home()
        result = {"success": True, "action": "go_home"}
    else:
        result = {"success": False, "error": f"未知模式: {mode}"}

    print(f"\n[Result] {result}")
    return result.get("success", False)


def main():
    parser = argparse.ArgumentParser(description="Piper + SAM3 抓取系统")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["grasp_simple", "grasp_servo",
                                 "estimate_pose", "search_and_grasp", "visualize_scene", "calibrate", "go_home"],
                        help="运行模式（不填则进入交互菜单）")
    parser.add_argument("--target",   type=str, default="apple",
                        help="目标物体 YOLO 类别名称（默认: apple）")
    parser.add_argument("--max_iter", type=int, default=20,
                        help="视觉伺服最大迭代次数（默认: 20）")
    parser.add_argument("--calib",    type=str,
                        default="/home/ubuntu/sunqianran/SAM3/calibration_result.npz",
                        help="手眼标定文件路径")
    args = parser.parse_args()

    rclpy.init()
    manager = GraspManager(calib_path=args.calib)

    success = True
    try:
        if args.mode is None:
            interactive_mode(manager)
        else:
            success = cli_mode(args, manager)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n[Error] {e}")
        traceback.print_exc()
        success = False
    finally:
        manager.shutdown()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
