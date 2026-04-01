import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import uvicorn

# 引入 Meta 官方 SAM3 原生运行库
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------------------------
# 命令行参数（在模块级解析，uvicorn 重载模式下也能正确读取）
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SAM3 分割服务器")
parser.add_argument("--vis", type=lambda x: x.lower() == "true",
                    default=False, metavar="True/False",
                    help="是否可视化推理结果（默认 False）")
parser.add_argument("--vis-dir", default="vis_output",
                    help="无显示器时保存可视化图片的目录（默认 vis_output/）")
parser.add_argument("--host", default="0.0.0.0",
                    help="监听地址（默认 0.0.0.0）")
parser.add_argument("--port", type=int, default=8000,
                    help="监听端口（默认 8000）")
# 忽略 uvicorn 自身可能注入的未知参数，避免解析报错
args, _ = parser.parse_known_args()

VIS_ENABLED = args.vis
VIS_DIR     = args.vis_dir

if VIS_ENABLED:
    print(f"[Server] 可视化已开启")
    has_display = bool(os.environ.get("DISPLAY", ""))
    if has_display:
        print("[Server] 检测到 DISPLAY，将使用 cv2.imshow 显示")
    else:
        os.makedirs(VIS_DIR, exist_ok=True)
        print(f"[Server] 未检测到 DISPLAY，可视化图片将保存至 {os.path.abspath(VIS_DIR)}/")

app = FastAPI()

print("[Server] 正在加载 SAM3 视觉大模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model_path = "sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam_model_path, device=device)
sam3_processor = Sam3Processor(sam3_model, device=device)
print("[Server] 模型加载完成，开始监听网络请求...")

# 用于给可视化图片编号（多请求并发时不会重名）
_vis_counter = 0


def _visualize_not_found(color_img: np.ndarray, target_class: str):
    """未检测到目标时，显示原图并在左上角标注红色提示。"""
    global _vis_counter

    vis = color_img.copy()
    label = f"NOT FOUND: {target_class}"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    has_display = bool(os.environ.get("DISPLAY", ""))
    if has_display:
        cv2.imshow(f"SAM3 - {target_class}", vis)
        cv2.waitKey(1)
    else:
        _vis_counter += 1
        fname = os.path.join(VIS_DIR, f"{_vis_counter:05d}_NOT_FOUND_{target_class}.jpg")
        cv2.imwrite(fname, vis)
        print(f"[Server] 可视化已保存: {fname}")


def _visualize(color_img: np.ndarray, mask_uint8: np.ndarray,
               target_class: str, score: float):
    """
    将原图与掩码叠加后显示或保存。
    - 掩码区域用半透明绿色覆盖
    - 左上角标注目标名称与置信度
    """
    global _vis_counter

    # 半透明绿色叠加
    overlay = color_img.copy()
    green_layer = np.zeros_like(color_img)
    green_layer[mask_uint8 > 0] = (0, 200, 0)   # BGR 绿色
    vis = cv2.addWeighted(overlay, 0.6, green_layer, 0.4, 0)

    # 掩码轮廓描边
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # 文字标注
    label = f"{target_class}  score={score:.3f}"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    has_display = bool(os.environ.get("DISPLAY", ""))
    if has_display:
        win = f"SAM3 - {target_class}"
        cv2.imshow(win, vis)
        cv2.waitKey(1)          # 非阻塞刷新，不影响服务器响应
    else:
        _vis_counter += 1
        fname = os.path.join(VIS_DIR, f"{_vis_counter:05d}_{target_class}.jpg")
        cv2.imwrite(fname, vis)
        print(f"[Server] 可视化已保存: {fname}")


@app.post("/predict")
async def predict(image: UploadFile = File(...), target_class: str = Form(...)):
    print(f"[Server] 收到预测请求，目标实体: {target_class}")

    # 1. 接收并解码 JPEG 字节流
    img_bytes = await image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    color_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if color_img is None:
        return Response(status_code=400, content="Invalid image format")

    # 2. 转换颜色空间并进入 SAM3 推理管线
    color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_rgb)

    state = sam3_processor.set_image(pil_img)
    state = sam3_processor.set_text_prompt(target_class, state)

    if "masks" not in state or len(state["masks"]) == 0:
        if VIS_ENABLED:
            _visualize_not_found(color_img, target_class)
        return Response(status_code=404, content="No mask found")

    masks = state["masks"]
    scores = state["scores"]

    masks_np  = masks.cpu().numpy()  if isinstance(masks,  torch.Tensor) else np.array(masks)
    scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)

    scores_np = scores_np.flatten()
    best_idx  = int(np.argmax(scores_np))
    if best_idx >= masks_np.shape[0]:
        best_idx = 0

    best_mask = np.squeeze(masks_np[best_idx])
    mask_uint8 = (best_mask > 0).astype(np.uint8) * 255

    # 3. 可视化（仅在 --vis True 时执行）
    if VIS_ENABLED:
        _visualize(color_img, mask_uint8, target_class, float(scores_np[best_idx]))

    # 4. 将 2D 二值掩码压缩为无损的 PNG 单通道字节流，极低带宽占用
    success, encoded_mask = cv2.imencode('.png', mask_uint8)
    if not success:
        return Response(status_code=500, content="Mask encode failed")

    return Response(content=encoded_mask.tobytes(), media_type="image/png")


if __name__ == "__main__":
    # host="0.0.0.0" 允许所有局域网设备访问
    uvicorn.run(app, host=args.host, port=args.port)
