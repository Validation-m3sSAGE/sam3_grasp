#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda create --name sam3_grasp python=3.10 -y
conda activate sam3_grasp
sudo apt update&&sudo apt install -y ethtool can-utils curl
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c
nvidia -y
pip install \ 
numpy scipy \ 
opencv-python \ 
open3d \ 
Pillow \ 
requests \ 
pyrealsense2 \ 
fastapi uvicorn \ 
mcp \ 
openai \ 
huggingface_hub \ 
iopath \ 
setuptools
pip install piper_sdk
pip install -e .
echo "✅ 安装完成。请将 sam3.pt 模型权重放置到项目根目录。"
echo "   可从 HuggingFace 下载: https://huggingface.co/facebook/sam3"
