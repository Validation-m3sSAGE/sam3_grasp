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

app = FastAPI()

print("[Server] 正在加载 SAM3 视觉大模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model_path = "sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam_model_path, device=device)
sam3_processor = Sam3Processor(sam3_model, device=device)
print("[Server] 模型加载完成，开始监听网络请求...")

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
        return Response(status_code=404, content="No mask found")

    masks = state["masks"]
    scores = state["scores"]
    
    masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
    scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
    
    scores_np = scores_np.flatten()
    best_idx = int(np.argmax(scores_np))
    if best_idx >= masks_np.shape[0]: 
        best_idx = 0
        
    best_mask = np.squeeze(masks_np[best_idx])
    mask_uint8 = (best_mask > 0).astype(np.uint8) * 255
    
    # 3. 将 2D 二值掩码压缩为无损的 PNG 单通道字节流，极低带宽占用
    success, encoded_mask = cv2.imencode('.png', mask_uint8)
    if not success:
        return Response(status_code=500, content="Mask encode failed")
        
    return Response(content=encoded_mask.tobytes(), media_type="image/png")

if __name__ == "__main__":
    # host="0.0.0.0" 允许所有局域网设备访问，默认绑定 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)