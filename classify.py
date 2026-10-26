import argparse
from pathlib import Path
from ultralytics import YOLO
parser = argparse.ArgumentParser(description="使用 GPU 进行单张图片分类")
parser.add_argument("image", help="图片路径")
args = parser.parse_args()
image_path = Path(args.image).expanduser()
model_path = Path(__file__).resolve().parent / "models" / "yolov8n-cls.pt"
if not image_path.is_file():
    parser.error(f"找不到图片：{image_path}")
if not model_path.is_file():
    parser.error(f"找不到模型：{model_path}")
model = YOLO(str(model_path))
result = model.predict(
    source=str(image_path),
    device=0,
    imgsz=224,
    verbose=False,
)[0]
print("图片：", image_path)
print("模型设备：", next(model.model.parameters()).device)
print("分类结果（前5名）：")
for index in result.probs.top5:
    score = result.probs.data[index].item()
    print(f"{result.names[index]}：{score:.2%}")
