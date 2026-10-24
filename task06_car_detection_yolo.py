"""Task 6: Car detection using YOLOv8 COCO pretrained weights.
Usage: python task06_car_detection_yolo.py data/car.jpg
"""
import sys
from pathlib import Path
from ultralytics import YOLO

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/car.jpg")
if not image_path.exists():
    raise SystemExit(f"Image not found: {image_path}. Supply an image path as an argument.")
model = YOLO("yolov8n.pt")  # Downloads weights on first run.
results = model.predict(source=str(image_path), conf=0.25, save=True, verbose=False)
names = model.names
count = 0
for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        if names[class_id] == "car":
            count += 1
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            print(f"Car {count}: box=({x1}, {y1}, {x2}, {y2}), confidence={float(box.conf.item()):.2f}")
print("Cars detected:", count)
print("Annotated image saved by Ultralytics in the runs/detect folder.")
