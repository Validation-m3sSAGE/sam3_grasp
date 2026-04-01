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
        print("  4. 旋转探索并释放 (explore_and_place)")
        print("  5. 旋转释放       (rotate_release)")
        print("  6. 旋转探索并抓取 (explore_and_grasp)")
        print("  7. 向右探索并释放 (explore_right_and_place)")
        print("  h. 机械臂回归标准俯视待机位姿")
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

        elif choice == "4":
            target_place = input("寻找放置目标 (默认 trash can): ").strip() or "trash can"
            res_place = manager.explore_and_place(target_place)
            print(f"探索放置结果: {res_place}")

        elif choice == "5":
            angle = input("旋转角度 (默认 60): ").strip()
            angle = float(angle) if angle else 60.0
            result = manager.rotate_and_release(angle_deg=angle)
            print(f"结果: {result}")

        elif choice == "h":
            result = manager.go_standby()
            print(f"结果: {{'success': {result}, 'action': 'go_standby'}}")

        elif choice == "6":
            target = input("寻找抓取目标 (默认 apple): ").strip() or "apple"
            res = manager.explore_and_grasp(target)
            print(f"结果: {res}")

        elif choice == "7":
            target_place = input("寻找放置目标 (默认 trash can): ").strip() or "trash can"
            res_place = manager.explore_right_and_place(target_place)
            print(f"探索向右放置结果: {res_place}")

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
        manager.go_standby()
        result = {"success": True, "action": "go_standby"}
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
