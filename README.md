# SAM3 × Piper 机械臂抓取系统

基于 Meta **SAM3**（Segment Anything Model 3）视觉大模型与松灵 **Piper** 机械臂的**零样本语言驱动抓取系统**。系统通过文本描述定位目标物体，结合 Intel RealSense D405 深度相机完成三维坐标解算，驱动机械臂执行精准抓取。系统已原生接入 **PhyAgentOS** 具身智能操作系统，并保留了对 **MCP（Model Context Protocol）** 的兼容支持。

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
│       │       └── HTTP → SAM3 推理服务 (server_sam3.py)     │
│       └── PiperArmControllerSDK (arm_controller_sdk.py)     │
│               ├── CAN 总线 (piper_sdk)                      │
│               └── OMPLKinematicsPlanner (ompl_planner.py)   │
│                       └── PyBullet IK/FK 引擎               │
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
| **手眼标定** | 多位姿 AX=XB（Tsai-Lenz / Park-Martin）手眼标定，自动计算相机系→机械臂基座系的变换矩阵 |
| **OMPL 运动规划** | 基于 PyBullet 物理引擎的 IK/FK 解算，将笛卡尔目标坐标映射为安全的关节空间轨迹 |
| **自动抓取** | 完整的预抓取→下放→夹取→复位动作链，支持 45° 斜向下抓取姿态 |
| **探索抓取** | 旋转扫描环境，发现目标后自动对准并执行抓取 |
| **PhyAgentOS 接入** | 原生支持作为 PhyAgentOS 开放模块接入，支持 Critic 安全校验与任务分解 |
| **MCP 工具接口** | 保留对 Model Context Protocol 的兼容，将抓取能力暴露为独立工具 |
| **Agentic 分割** | 内置多轮 MLLM 推理 Agent，支持复杂指代表达的精确分割 |

---

## 目录结构

```
SAM3/
├── phyagentos_driver.py       # PhyAgentOS 驱动实现（核心接入点）
├── phyagentos_bridge_server.py# PhyAgentOS 远程模式 HTTP 桥接服务
├── PhyAgentOS_plugin.toml     # PhyAgentOS 插件注册配置
├── register_to_phyagentos.py  # 快捷注册脚本
├── profiles/                  # 机器人能力描述文件
│   └── sam3_piper.md          # 供 Critic 校验的 EMBODIED.md 模板
├── main.py                    # 主入口（交互菜单 / CLI 模式）
├── mcp_server.py              # MCP 服务端（供 AI Agent 调用）
├── server_sam3.py             # SAM3 GPU 推理 HTTP 服务
├── grasp_manager.py           # 抓取管理器（标定、感知、运动规划）
├── perception_layer.py        # 感知层（相机采集、掩码提取、点云处理）
├── arm_controller_sdk.py      # 机械臂控制器（基于 piper_sdk + OMPL）
├── ompl_planner.py            # PyBullet IK/FK 运动学规划器
├── launch_sam3_system.sh      # 一键启动脚本（CAN + 推理服务 + MCP）
├── can_activate.sh            # CAN 总线激活脚本
├── install.sh                 # 环境安装脚本
├── setup.py                   # sam3 子包安装配置
├── calibration_result.npz     # 手眼标定矩阵（运行后自动生成，不纳入版本控制）
├── sam3.pt                    # SAM3 模型权重（需手动下载，不纳入版本控制）
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
├── urdf/                      # Piper 机械臂 URDF 模型（供 PyBullet IK 使用）
│   └── piper_description.urdf
│
└── ros__fixed_long_ver/       # 历史 ROS2 版本（已弃用，仅供参考）
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone <repo_url>
cd SAM3
```

### 2. 环境安装

```bash
bash install.sh
```

> **前提**：已安装 [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html) 和 NVIDIA GPU 驱动。

安装完成后，手动下载模型权重并放置到项目根目录：

```bash
# 从 HuggingFace 下载
huggingface-cli download facebook/sam3 sam3.pt --local-dir .
# 或手动复制
cp /path/to/sam3.pt ./sam3.pt
```

### 3. 配置感知层服务器地址

若推理服务（`server_sam3.py`）与控制端**不在同一台机器**，编辑 `perception_layer.py`：

```python
# perception_layer.py 第 16 行
server_ip: str = "127.0.0.1",   # ← 修改为运行 server_sam3.py 的 GPU 主机局域网 IP
```

### 4. 启动推理服务（GPU 主机）

```bash
conda activate sam3_grasp
python server_sam3.py
# 服务将在 http://0.0.0.0:8000 监听

# 可选：开启可视化（无显示器时自动保存到 vis_output/）
python server_sam3.py --vis True
```

### 5a. 直接运行（交互模式）

```bash
conda activate sam3_grasp
bash can_activate.sh can0 1000000   # 激活 CAN 总线
python main.py
```

交互菜单选项：

```
1. 单次直接抓取   (grasp_simple)        — 输入目标名称，执行完整抓取
2. 可视化当前点云 (visualize_scene)      — 保存当前场景点云到 .ply 文件
3. 执行手眼标定   (calibrate)           — AX=XB 多位姿手眼标定
4. 旋转探索并释放 (explore_and_place)   — 向左旋转寻找目标，在目标上方释放
5. 旋转释放       (rotate_release)      — 旋转指定角度后释放夹持物体
6. 旋转探索并抓取 (explore_and_grasp)   — 向右旋转寻找目标，发现后执行抓取
7. 向右探索并释放 (explore_right_and_place) — 向右旋转寻找目标，在目标上方释放
h. 机械臂回归标准俯视待机位姿
q. 退出
```

### 5b. CLI 模式

```bash
python main.py --mode grasp_simple --target apple
python main.py --mode calibrate
python main.py --mode go_home
python main.py --mode visualize_scene
python main.py --calib /path/to/calibration_result.npz --mode grasp_simple --target bottle
```

### 5c. PhyAgentOS 模式（推荐）

SAM3 系统已原生支持作为 **PhyAgentOS 开放模块（External Plugin Driver）** 接入。接入后，PhyAgentOS 的 AgentLoop 可以直接控制 SAM3 + Piper 系统执行抓取任务，并支持任务分解、Critic 安全校验和状态持久化。

#### 注册插件（一次性）

在 PhyAgentOS 环境中执行以下命令，将 SAM3 注册为驱动插件：

```bash
cd /path/to/PhyAgentOS
python -c "
from hal.plugins import register_plugin
spec = register_plugin('/path/to/SAM3')
print('Plugin registered:', spec.driver_name)
"
```

> **提示**：如果你觉得命令行注册比较麻烦，也可以直接运行 SAM3 目录下的 `register_to_phyagentos.py` 脚本（需指定 PhyAgentOS 的路径）。

#### 启动方式

**方式一：本地模式（PhyAgentOS 与 SAM3 在同一台机器）**

```bash
conda activate sam3_grasp
python -m hal.hal_watchdog \
    --driver sam3_piper \
    --workspace ~/.PhyAgentOS/workspace \
    --interval 1.0
```

**方式二：远程模式（PhyAgentOS 在控制主机，SAM3 在机械臂机器）**

1. 在机械臂机器上启动桥接服务：
   ```bash
   conda activate sam3_grasp
   python phyagentos_bridge_server.py --port 18791
   ```

2. 在控制主机上创建 `sam3_remote.json`：
   ```json
   {
     "mode": "remote",
     "bridge_url": "http://<机械臂机器IP>:18791"
   }
   ```

3. 在控制主机上启动 Watchdog：
   ```bash
   python -m hal.hal_watchdog \
       --driver sam3_piper \
       --driver-config /path/to/sam3_remote.json \
       --workspace ~/.PhyAgentOS/workspace
   ```

在支持的 AI Agent（如 OEA）中，可通过自然语言指令控制机械臂：

> "帮我抓取桌上的苹果"  
> "检查视野中是否有水瓶"  
> "执行手眼标定"  
> "把夹着的东西放到垃圾桶里"

#### 支持的动作 (Action Types)

| 动作名 | 参数 | 功能 |
|--------|------|------|
| `grasp` | `target_name: str` | 直接抓取指定目标物体 |
| `explore_and_grasp` | `target_name: str` | 向右旋转扫描，发现目标后执行抓取 |
| `explore_and_place` | `target_name: str` | 向左旋转扫描，发现目标后在其上方释放 |
| `explore_right_and_place` | `target_name: str` | 向右旋转扫描，发现目标后在其上方释放 |
| `check_target` | `target_name: str` | 检查指定目标是否在当前视野中 |
| `calibrate` | — | 执行多位姿 AX=XB 手眼标定 |
| `go_home` | — | 机械臂回归标准俯视待机位姿 |
| `connect` / `disconnect` | — | 使能/关闭机械臂 |

### 5d. MCP Agent 模式（旧版兼容）

系统仍保留了对 MCP（Model Context Protocol）的支持。通过 `launch_sam3_system.sh` 可一键启动完整系统（CAN 激活 + 推理服务 + MCP 服务端）。

---

## 手眼标定

首次使用或抓取位置出现系统性偏移时，需执行手眼标定。

**标定原理**：采用 **AX=XB（Park-Martin 非线性逼近）** 多位姿手眼标定算法（眼在手上，Eye-in-Hand）。机械臂在初始观测位姿附近自动采集 15 组随机扰动位姿，每个位姿下通过 OpenCV `solvePnP` 解算棋盘格在相机系下的 6D 位姿，同时读取机械臂正运动学末端矩阵，最终通过 `cv2.calibrateHandEye(method=CALIB_HAND_EYE_PARK)` 求解相机系→末端法兰系的变换矩阵 `T_ee_to_cam`。

**标定要求**：
- 准备一块 **9×12 格子（内角点 11×8）、方格边长 10mm** 的棋盘格标定板
- 将标定板平放在机械臂正下方工作区域中心
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
                                         T_ee_to_cam 变换
                                                │
                                         3D 坐标 (基座系)
                                                │
                                         PyBullet IK 规划
                                                │
                                         机械臂关节控制
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
| `opencv-python` | 图像编解码、掩码处理、手眼标定 |
| `open3d` | 点云处理、可视化 |
| `pybullet` | PyBullet 物理引擎（IK/FK 运动学规划） |
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
6. **模型权重与标定文件**：`sam3.pt` 和 `calibration_result.npz` 均已加入 `.gitignore`，不会被提交到版本控制。

---

## 许可证

SAM3 模型代码遵循 Meta Platforms 版权声明（见各源文件头部）。本项目的机器人控制与感知集成代码为独立开发。
