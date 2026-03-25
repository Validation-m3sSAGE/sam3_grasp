# SAM3 × Piper 机械臂抓取系统

基于 Meta SAM3（Segment Anything Model 3）视觉大模型与松灵 Piper 机械臂的**零样本语言驱动抓取系统**。系统通过文本描述定位目标物体，结合 Intel RealSense D405 深度相机完成三维坐标解算，驱动机械臂执行精准抓取。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        控制端（边缘设备）                      │
│                                                             │
│  main.py / mcp_server.py                                    │
│       │                                                     │
│  GraspManager (grasp_manager.py)                            │
│       ├── PerceptionLayer (perception_layer.py)             │
│       │       ├── RealSense D405 (pyrealsense2)             │
│       │       └── HTTP → GPU 推理服务 (server_sam3.py)       │
│       └── PiperArmControllerSDK (arm_controller_sdk.py)     │
│               └── CAN 总线 (piper_sdk)                      │
└─────────────────────────────────────────────────────────────┘
                              │ HTTP :8000
┌─────────────────────────────────────────────────────────────┐
│                     推理端（GPU 主机）                         │
│                                                             │
│  server_sam3.py  (FastAPI + uvicorn)                        │
│       └── SAM3 模型 (sam3.pt)                               │
│               ├── ViT-L 视觉骨干网络                          │
│               ├── 文本编码器 (CLIP-style BPE)                 │
│               └── 分割头 + Transformer 解码器                 │
└─────────────────────────────────────────────────────────────┘
```

> **部署说明**：推理端与控制端可以是同一台机器（`server_ip=127.0.0.1`），也可以分离部署（边缘设备 + GPU 服务器）。

---

## 核心功能

| 功能 | 描述 |
|------|------|
| **零样本分割** | 输入任意英文名词（如 `apple`、`bottle`），SAM3 自动定位并生成像素级掩码 |
| **3D 坐标解算** | 将 2D 掩码与 D405 深度图融合，计算目标物体在相机坐标系下的三维重心 |
| **手眼标定** | 双向多点 ICP 点云配准，自动计算相机系→机械臂基座系的旋转映射矩阵 |
| **自动抓取** | 完整的预抓取→下放→夹取→复位动作链 |
| **MCP 工具接口** | 通过 Model Context Protocol 将抓取能力暴露为 AI Agent 可调用的工具 |
| **Agentic 分割** | 内置多轮 MLLM 推理 Agent，支持复杂指代表达的精确分割 |

---

## 目录结构

```
SAM3/
├── main.py                    # 主入口（交互菜单 / CLI 模式）
├── mcp_server.py              # MCP 服务端（供 AI Agent 调用）
├── server_sam3.py             # SAM3 GPU 推理 HTTP 服务
├── grasp_manager.py           # 抓取管理器（标定、感知、运动规划）
├── perception_layer.py        # 感知层（相机采集、掩码提取、点云处理）
├── arm_controller_sdk.py      # 机械臂控制器（基于 piper_sdk）
├── launch_sam3_system.sh      # 一键启动脚本（CAN + 推理服务 + MCP）
├── can_activate.sh            # CAN 总线激活脚本
├── install.sh                 # 环境安装脚本
├── calibration_result.npz     # 手眼标定矩阵（持久化存储）
├── sam3.pt                    # SAM3 模型权重（需手动下载）
│
├── sam3/                      # SAM3 模型库（Meta 官方代码）
│   ├── model_builder.py       # 模型构建入口
│   ├── model/                 # 模型核心组件
│   │   ├── sam3_image.py      # 图像推理模型
│   │   ├── sam3_image_processor.py  # 推理处理器（set_image / set_text_prompt）
│   │   ├── sam3_video_predictor.py  # 视频追踪预测器
│   │   ├── encoder.py / decoder.py  # Transformer 编解码器
│   │   ├── vitdet.py          # ViT 视觉骨干
│   │   └── ...
│   ├── agent/                 # Agentic 分割模块
│   │   ├── agent_core.py      # 多轮 MLLM 推理 Agent 主逻辑
│   │   ├── client_llm.py      # LLM 客户端（OpenAI 兼容接口）
│   │   ├── client_sam3.py     # SAM3 服务调用客户端
│   │   ├── system_prompts/    # Agent 系统提示词
│   │   └── helpers/           # 可视化、掩码处理工具
│   ├── eval/                  # 评测套件（COCO、HOTA、TETA 等）
│   ├── train/                 # 训练框架（数据加载、损失函数、优化器）
│   ├── perflib/               # 高性能算子（Triton NMS、连通域等）
│   └── sam/                   # SAM1 兼容组件
│
└── ros__fixed_long_ver/       # 历史 ROS2 版本（已弃用，仅供参考）
```

---

## 快速开始

### 1. 环境安装

```bash
bash install.sh
```

安装完成后，手动下载模型权重并放置到项目根目录：

```bash
# 方式一：从 HuggingFace 下载（代码中已内置自动下载逻辑）
# 方式二：手动下载后放置
cp /path/to/sam3.pt /home/ubuntu/sunqianran/SAM3/sam3.pt
```

### 2. 配置感知层服务器地址

编辑 `perception_layer.py`，将 `server_ip` 修改为运行 `server_sam3.py` 的 GPU 主机 IP：

```python
# perception_layer.py 第 15 行
server_ip: str = "127.0.0.1",   # ← 修改为 GPU 主机的局域网 IP
```

### 3. 启动推理服务（GPU 主机）

```bash
conda activate sam3_grasp
python server_sam3.py
# 服务将在 http://0.0.0.0:8000 监听
```

### 4a. 直接运行（交互模式）

```bash
conda activate sam3_grasp
bash can_activate.sh can0 1000000   # 激活 CAN 总线
python main.py
```

交互菜单选项：

```
1. 单次直接抓取   (grasp_simple)   — 输入目标名称，执行完整抓取
2. 可视化当前点云 (visualize_scene) — Open3D 窗口渲染当前场景
3. 执行手眼标定   (calibrate)      — 双向 ICP 全轴标定
h. 机械臂回零
q. 退出
```

### 4b. CLI 模式

```bash
python main.py --mode grasp_simple --target apple
python main.py --mode calibrate
python main.py --mode go_home
python main.py --mode visualize_scene
python main.py --calib /path/to/calibration_result.npz --mode grasp_simple --target bottle
```

### 4c. MCP Agent 模式（推荐）

通过 `launch_sam3_system.sh` 一键启动完整系统（CAN 激活 + 推理服务 + MCP 服务端）：

```bash
bash launch_sam3_system.sh
```

该脚本会：
1. 激活 CAN 总线
2. 后台启动 `server_sam3.py`（等待 HTTP 服务就绪）
3. 前台启动 `mcp_server.py`（接管 stdio，供 AI Agent 调用）

在 OEA 或其他支持 MCP 的 AI Agent 中配置此工具后，可通过自然语言指令控制机械臂：

> "帮我抓取桌上的苹果"  
> "检查视野中是否有水瓶"  
> "执行手眼标定"

---

## 手眼标定

首次使用或抓取位置出现系统性偏移时，需执行手眼标定。

**标定原理**：机械臂沿 X/Y/Z 三轴各做正负方向微动（默认 ±5cm），通过 Open3D ICP 点云配准计算相机坐标系下的位移向量，SVD 正交化后得到相机系→基座系的旋转矩阵 `R_cam2base`。

**标定要求**：
- 相机视野内放置几何特征丰富的物体（非平面、有棱角）
- 确保机械臂运动范围内无障碍物
- 标定结果自动保存至 `calibration_result.npz`，下次启动自动加载

```bash
python main.py --mode calibrate
```

---

## 感知管线详解

```
RealSense D405
    │
    ├── 彩色帧 (BGR, 848×480)
    │       │
    │       └── JPEG 压缩 (quality=85) → HTTP POST → server_sam3.py
    │                                           │
    │                                    SAM3 文本分割
    │                                           │
    │                                    PNG 掩码回传
    │
    └── 深度帧 (Z16, 对齐到彩色)
            │
            └── 与掩码融合 → 点云重心计算 → 3D 坐标 (相机系)
                                                │
                                         R_cam2base 变换
                                                │
                                         3D 坐标 (基座系)
                                                │
                                         机械臂运动规划
```

---

## Agentic 分割（`sam3/agent/`）

对于复杂的指代表达（如"桌子左边第二个红色杯子"），系统内置了基于多模态大语言模型（MLLM）的多轮推理 Agent：

- **LLM 后端**：兼容 OpenAI API 格式，默认使用 `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`，可替换为任意兼容模型
- **工具集**：`segment_phrase`、`examine_each_mask`、`select_masks_and_return`、`report_no_mask`
- **调用方式**：

```python
from sam3.agent.agent_core import agent_inference

messages, final_outputs, rendered_image = agent_inference(
    img_path="path/to/image.jpg",
    initial_text_prompt="the red apple on the left",
)
```

---

## 硬件要求

| 组件 | 规格 |
|------|------|
| 机械臂 | 松灵 Piper（6 自由度，CAN 总线通信） |
| 深度相机 | Intel RealSense D405（最小工作距离 70mm，最大 1000mm） |
| GPU 主机 | NVIDIA GPU（推荐 ≥ 8GB 显存，用于 SAM3 推理） |
| 边缘控制端 | 任意 Linux 主机，需 USB-CAN 适配器 |
| 操作系统 | Ubuntu 20.04 / 22.04 |

---

## 依赖说明

| 包 | 用途 |
|----|------|
| `torch` / `torchvision` | SAM3 模型推理 |
| `numpy` | 数值计算、坐标变换 |
| `scipy` | 旋转矩阵计算（`Rotation.from_euler`） |
| `opencv-python` | 图像编解码、掩码处理 |
| `open3d` | 点云处理、ICP 配准、可视化 |
| `pyrealsense2` | RealSense D405 相机驱动 |
| `requests` | 边缘端→GPU 主机 HTTP 通信 |
| `fastapi` + `uvicorn` | SAM3 推理 HTTP 服务 |
| `mcp` | Model Context Protocol 服务端 |
| `piper_sdk` | 松灵 Piper 机械臂 CAN 通信 SDK |
| `Pillow` | 图像格式转换 |
| `openai` | Agentic 分割 LLM 客户端 |
| `iopath` | 模型权重文件 I/O |
| `huggingface_hub` | 从 HuggingFace 自动下载模型权重 |

---

## 注意事项

1. **首次运行必须先标定**：若 `calibration_result.npz` 不存在，抓取功能将拒绝执行。
2. **CAN 总线需要 sudo 权限**：`can_activate.sh` 内部使用 `sudo ip link` 命令，请确保当前用户有相应权限。
3. **server_ip 配置**：`perception_layer.py` 中的 `server_ip` 默认为 `127.0.0.1`，若推理服务部署在远程 GPU 主机，需修改为对应 IP。
4. **工作空间限制**：机械臂有效工作半径约 0.45m，高度范围 -0.05m ~ 0.40m，超出范围会触发警告（系统仍会尝试执行）。
5. **`ros__fixed_long_ver/` 目录**：为早期 ROS2 版本的历史存档，当前系统已完全迁移至 piper_sdk 直驱模式，无需 ROS 环境。

---

## 许可证

SAM3 模型代码遵循 Meta Platforms 版权声明（见各源文件头部）。本项目的机器人控制与感知集成代码为独立开发。
