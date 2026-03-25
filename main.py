#!/usr/bin/env python3
"""
松灵 Piper + SAM3 抓取系统 (SDK 专版) 入口
"""

import argparse
import sys
import traceback
from grasp_manager import GraspManager


def interactive_mode(manager):
    while True:
        print()
        print("=" * 50)
        print("  松灵Piper + SAM3 抓取系统 (SDK专版)")
        print("=" * 50)
        print("  1. 单次直接抓取   (grasp_simple)")
        print("  2. 可视化当前点云 (visualize_scene)")
        print("  3. 执行手眼标定   (calibrate)")
        print("  h. 机械臂回零")
        print("  q. 退出")
        print("-" * 50)

        choice = input("输入选项: ").strip().lower()

        if choice == "1":
            target = input("目标名称 (默认 apple): ").strip() or "apple"
            result = manager.grasp_simple(target)
            print(f"结果: {result}")

        elif choice == "2":
            result = manager.visualize_scene()
            print(f"结果: {result}")

        elif choice == "3":
            try:
                manager.calibrate_axes()
                print("结果: {'success': True}")
            except Exception as e:
                print(f"结果: {{'success': False, 'error': '{str(e)}'}}")

        elif choice == "h":
            result = manager.arm.go_home()
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
    parser = argparse.ArgumentParser(description="Piper + SAM3 抓取系统 (SDK 专版)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["grasp_simple", "visualize_scene", "calibrate", "go_home"],
                        help="运行模式（不填则进入交互菜单）")
    parser.add_argument("--target",   type=str, default="apple",
                        help="目标物体类别名称（默认: apple）")
    parser.add_argument("--calib",    type=str,
                        default="calibration_result.npz",
                        help="手眼标定文件路径")
    args = parser.parse_args()

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

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
