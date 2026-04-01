#!/bin/bash
set -e

# ============================================================
# SAM3 × Piper 抓取系统 — 一键环境安装脚本
# 适用于 Ubuntu 20.04 / 22.04，需要 Conda 和 NVIDIA GPU
# ============================================================

eval "$(conda shell.bash hook)"

# 1. 创建并激活 Conda 虚拟环境
conda create --name sam3_grasp python=3.10 -y
conda activate sam3_grasp

# 2. 安装系统依赖（CAN 总线工具）
sudo apt update && sudo apt install -y ethtool can-utils curl

# 3. 安装 PyTorch（CUDA 12.1）
#    如需其他 CUDA 版本，请参考 https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y

# 4. 安装 Python 依赖
pip install \
    numpy \
    scipy \
    opencv-python \
    open3d \
    Pillow \
    requests \
    pyrealsense2 \
    fastapi \
    uvicorn \
    mcp \
    openai \
    huggingface_hub \
    iopath \
    pybullet \
    setuptools

# 5. 安装松灵 Piper 机械臂 SDK
pip install piper_sdk

# 6. 以可编辑模式安装 SAM3 模型库（sam3/ 子包）
pip install -e .

echo ""
echo "✅ 安装完成！"
echo ""
echo "后续步骤："
echo "  1. 下载 SAM3 模型权重并放置到项目根目录："
echo "     cp /path/to/sam3.pt $(pwd)/sam3.pt"
echo "     （或从 HuggingFace 下载：https://huggingface.co/facebook/sam3）"
echo ""
echo "  2. 若推理服务部署在远程 GPU 主机，修改 perception_layer.py 中的 server_ip 参数。"
echo ""
echo "  3. 首次运行前执行手眼标定："
echo "     conda activate sam3_grasp && python main.py --mode calibrate"
