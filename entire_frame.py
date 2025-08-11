# detect_people.py
import cv2
from ultralytics import YOLO

# 1) 경로 설정 --------------------------------------------------------
IMG_PATH   = 'subway.jpg'   # 탐지할 원본 사진
MODEL_PATH = 'yolov8n-pose.pt'       # 기본 사람 클래스 포함

# 2) 모델 로드 --------------------------------------------------------
model = YOLO(MODEL_PATH)

# 3) 사람 탐지 --------------------------------------------------------
img     = cv2.imread(IMG_PATH)
results = model(img, classes=[0], conf=0.3, verbose=False)  # class 0 = person
boxes   = results[0].boxes  # Boxes object (xyxy, conf, cls)

# 4) 박스 그리기 ------------------------------------------------------
vis = img.copy()
for xyxy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f'{conf:.2f}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

# 5) 결과 저장/표시 ----------------------------------------------------
cv2.imwrite('people_detect1.jpg', vis)
cv2.imshow('People Detection', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('완료: people_detect.jpg 저장')
