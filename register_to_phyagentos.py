#!/usr/bin/env python3
"""
SAM3 快捷注册脚本 (注册到 PhyAgentOS)

此脚本用于将当前 SAM3 目录作为外部插件注册到 PhyAgentOS 中。
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="将 SAM3 注册到 PhyAgentOS")
    parser.add_argument(
        "--phyagentos-path", 
        type=str, 
        help="PhyAgentOS 项目的绝对路径 (例如: /home/ubuntu/PhyAgentOS)",
        required=True
    )
    args = parser.parse_args()

    phy_path = Path(args.phyagentos_path).expanduser().resolve()
    if not (phy_path / "hal" / "plugins.py").exists():
        print(f"错误: 在 {phy_path} 下未找到 PhyAgentOS 核心文件 (hal/plugins.py)。")
        print("请确保提供了正确的 PhyAgentOS 项目路径。")
        sys.exit(1)

    # 将 PhyAgentOS 加入 sys.path 以便导入其模块
    sys.path.insert(0, str(phy_path))

    try:
        from hal.plugins import register_plugin
    except ImportError as e:
        print(f"错误: 无法从 PhyAgentOS 导入插件注册模块: {e}")
        sys.exit(1)

    sam3_path = Path(__file__).parent.resolve()
    
    print(f"正在将 SAM3 ({sam3_path}) 注册到 PhyAgentOS ({phy_path})...")
    
    try:
        spec = register_plugin(sam3_path)
        print("
✅ 注册成功！")
        print(f"  驱动名称: {spec.driver_name}")
        print(f"  插件名称: {spec.plugin_name}")
        print(f"  配置文件: {spec.profile_path}")
        print("
现在你可以使用以下命令启动 Watchdog:")
        print(f"  python -m hal.hal_watchdog --driver {spec.driver_name} --workspace ~/.PhyAgentOS/workspace --interval 1.0")
    except Exception as e:
        print(f"
❌ 注册失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
