eval "$(conda shell.bash hook)"
conda create --name sam3_grasp python=3.10 -y
conda activate sam3_grasp
conda install pytorch torchvision torchaudio pytorch-cuda=latest -c pytorch -c nvidia -y
pip install numpy scipy opencv-python open3d mcp pyrealsense2 requests piper_sdk
echo "Please download sam3.pt and place it here."