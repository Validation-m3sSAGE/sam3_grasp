import json
import os
import sys
import builtins
from datetime import datetime

# 重载全局 print 函数，将所有标准输出强行导向标准错误流
# 该机制为零侵入防御，可防止底层逻辑库中的大量 print 语句破坏 MCP 协议标准输入输出管道
original_print = builtins.print
def print_to_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    original_print(*args, **kwargs)
builtins.print = print_to_stderr

from mcp.server.fastmcp import FastMCP

# 引入现有的硬件与感知闭环管理类
from grasp_manager import GraspManager

# 1. 初始化 MCP 服务端节点
mcp = FastMCP("SAM3_Piper_Grasp_Node")

# 2. 全局实例化单例，避免重复启动硬件通信
manager = GraspManager(calib_path="calibration_result.npz")
LOG_FILE = "mcp_interaction_log.jsonl"

def append_to_log(role: str, content: str):
    """
    底层独立的日志沉淀逻辑，将文本交互记录写入统一文件。
    该方法强制执行，不依赖上层 LLM 的意图。
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": str(content)
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# 3. 工具注册：通过类型注解定义入参 Schema

@mcp.tool()
def grasp_simple(target_name: str = "apple") -> str:
    """
    调用 SAM3 视觉模型和 Piper 机械臂执行单次抓取。
    入参 target_name 是想要抓取的物体英文名称。
    """
    append_to_log("assistant", f"调用抓取指令: {target_name}")
    try:
        result = manager.grasp_simple(target_name)
        res_text = f"抓取完成，系统返回状态: {result}"
        append_to_log("sam3_grasp_piper", res_text)
        return res_text
    except Exception as e:
        err_text = f"抓取过程发生硬件或感知异常: {str(e)}"
        append_to_log("sam3_grasp_piper", err_text)
        return err_text

@mcp.tool()
def explore_and_place(target_name: str = "trash can") -> str:
    """
    在机械臂已夹持物体的情况下，旋转寻找指定目标，并在目标上方释放。
    入参 target_name 是想要寻找的投放目标英文名称（如 trash can）。
    """
    append_to_log("assistant", f"调用探索放置指令: {target_name}")
    try:
        result = manager.explore_and_place(target_name)
        res_text = f"探索与放置完成，系统返回状态: {result}"
        append_to_log("sam3_grasp_piper", res_text)
        return res_text
    except Exception as e:
        err_text = f"探索放置过程发生硬件或感知异常: {str(e)}"
        append_to_log("sam3_grasp_piper", err_text)
        return err_text

@mcp.tool()
def explore_and_grasp(target_name: str = "apple") -> str:
    """
    在视野中未见目标时，强制向右旋转扫描寻找指定物体。一旦找到并对准中心，立即执行抓取。
    入参 target_name 是想要寻找的抓取目标英文名称。
    """
    append_to_log("assistant", f"调用向右探索抓取指令: {target_name}")
    try:
        result = manager.explore_and_grasp(target_name)
        res_text = f"探索与抓取完成，系统返回状态: {result}"
        append_to_log("sam3_grasp_piper", res_text)
        return res_text
    except Exception as e:
        err_text = f"探索抓取过程发生硬件或感知异常: {str(e)}"
        append_to_log("sam3_grasp_piper", err_text)
        return err_text

@mcp.tool()
def explore_right_and_place(target_name: str = "trash can") -> str:
    """
    在机械臂已夹持物体的情况下，向右旋转寻找指定目标，并在目标上方释放。
    入参 target_name 是想要寻找的投放目标英文名称（如 trash can）。
    """
    append_to_log("assistant", f"调用向右探索放置指令: {target_name}")
    try:
        result = manager.explore_right_and_place(target_name)
        res_text = f"向右探索与放置完成，系统返回状态: {result}"
        append_to_log("sam3_grasp_piper", res_text)
        return res_text
    except Exception as e:
        err_text = f"向右探索放置过程发生硬件或感知异常: {str(e)}"
        append_to_log("sam3_grasp_piper", err_text)
        return err_text

@mcp.tool()
def check_target(target_name: str = "apple") -> str:
    """
    检查指定目标物体当前是否在相机视野中。
    入参 target_name 是想要检查的物体英文名称。
    """
    append_to_log("assistant", f"调用目标检测指令: {target_name}")
    try:
        result = manager.check_target_in_scene(target_name)
        if result.get('found', False):
            res_text = f"目标 '{target_name}' 已在视野中找到。"
        else:
            res_text = f"目标 '{target_name}' 未在视野中找到。"
        append_to_log("sam3_grasp_piper", res_text)
        return res_text
    except Exception as e:
        err_text = f"检测过程发生感知异常: {str(e)}"
        append_to_log("sam3_grasp_piper", err_text)
        return err_text

@mcp.tool()
def calibrate_axes() -> str:
    """
    执行机械臂与相机的双向高精度全轴手眼标定。
    默认标定文件已存在，当抓取发生严重位置偏移时方可调用此工具。
    请向用户确认，保证标定板已放置妥当。
    """
    append_to_log("assistant", "调用手眼标定指令")
    try:
        manager.calibrate_axes()
        res_text = "全轴标定完成，坐标映射矩阵已成功持久化。"
        append_to_log("sam3_grasp_piper", res_text)
        return res_text
    except Exception as e:
        err_text = f"标定失败: {str(e)}"
        append_to_log("sam3_grasp_piper", err_text)
        return err_text

@mcp.tool()
def go_home() -> str:
    """
    令机械臂强制返回待机观测原点 (高空俯视)。
    """
    append_to_log("assistant", "调用机械臂回归待机指令")
    manager.go_standby()
    res_text = "已下发回归标准待机位姿动作指令。"
    append_to_log("sam3_grasp_piper", res_text)
    return res_text

if __name__ == "__main__":
    # 使用 stdio 传输层启动服务，拦截并接管标准输入输出以匹配 OEA 调用
    mcp.run(transport="stdio")
