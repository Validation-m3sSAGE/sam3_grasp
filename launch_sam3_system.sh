#!/bin/bash
# ============================================================
# SAM3 × Piper 抓取系统 — 一键启动脚本
# 功能：激活 CAN 总线 → 启动 SAM3 推理服务 → 前台运行 MCP Server
# ============================================================

# 定义清理函数，当脚本收到退出信号时执行
cleanup() {
    echo "收到终止信号，正在清理子进程..." >&2
    if [ -n "$LOCAL_PID" ]; then
        kill "$LOCAL_PID" 2>/dev/null
    fi
    pkill -f "server_sam3.py" 2>/dev/null
    exit 0
}

# 绑定 SIGTERM 和 SIGINT 信号到 cleanup 函数
trap cleanup SIGTERM SIGINT

# 切换到脚本所在目录（兼容任意部署路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 在启动前释放可能被残留进程占用的 8000 端口
pkill -f "server_sam3.py" 2>/dev/null
sleep 0.5

# 1. 激活 CAN 总线（输出重定向到 stderr，防止污染 MCP 通信流）
bash can_activate.sh can0 1000000 >&2

# 2. 激活 Conda 环境
eval "$(conda shell.bash hook)"
conda activate sam3_grasp

# 3. 后台启动 SAM3 GPU 推理服务
python3 server_sam3.py >&2 &
LOCAL_PID=$!

# 4. 等待 server_sam3.py 的 HTTP 服务在 8000 端口就绪（最多等 60 秒）
echo "等待 SAM3 推理服务就绪 (http://127.0.0.1:8000)..." >&2
WAIT_SECS=0
until curl -sf http://127.0.0.1:8000/docs > /dev/null 2>&1; do
    sleep 1
    WAIT_SECS=$((WAIT_SECS + 1))
    if [ "$WAIT_SECS" -ge 60 ]; then
        echo "错误：SAM3 推理服务 60 秒内未就绪，终止启动。" >&2
        kill "$LOCAL_PID" 2>/dev/null
        exit 1
    fi
done
echo "SAM3 推理服务已就绪（等待了 ${WAIT_SECS} 秒）。" >&2

# 5. 前台启动主 MCP Server，接管标准输入输出流
#    注意：必须阻塞执行，不能添加后台运行符 &
python3 mcp_server.py
