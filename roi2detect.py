import cv2, json, numpy as np
from ultralytics import YOLO

# --- 모델 로드 : .pt 또는 .onnx ---
model = YOLO('yolov8n.pt')   # 또는 'yolov8n.onnx'

# --- 다각형 ROI 불러오기 ---
rois = {k: np.array(v, np.float32)
        for k, v in json.load(open('seat_roi.json')).items()}

def poly_iou(poly, box):
    bx1, by1, bx2, by2 = box
    rect = np.array([[bx1,by1],[bx2,by1],[bx2,by2],[bx1,by2]], np.int32)
    inter = cv2.intersectConvexConvex(poly.astype(np.float32),
                                      rect.astype(np.float32))[0]
    if inter <= 0: return 0.
    area_p = cv2.contourArea(poly); area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_p + area_b - inter + 1e-6)

frame = cv2.imread('subway.jpg')
people = model(frame, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()

for sid, poly in rois.items():
    occupied = any(poly_iou(poly, p) > 0.25 for p in people)
    print(f'Seat {sid}:', 'Occupied' if occupied else 'Empty')
