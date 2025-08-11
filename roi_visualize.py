import cv2, json, numpy as np
from pathlib import Path

# ---------- 경로 설정 ----------
IMG_PATH  = 'subway2.jpg'          # 원본 이미지
JSON_PATH = 'seat_roi2.json'           # 질문에 첨부한 내용 저장
OUT_PATH  = 'vis_rois2.jpg'            # 시각화 결과

# ---------- 데이터 로드 ----------
img  = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f'이미지를 찾을 수 없습니다: {IMG_PATH}')

rois = json.load(open(JSON_PATH, 'r'))   # dict[str, list[list[int,int]]]

# ---------- ROI 시각화 ----------
canvas = img.copy()
for sid, pts in rois.items():
    quad = np.array(pts, dtype=np.int32)

    # 1) 폴리곤(사다리꼴·평행사변형 포함)
    cv2.polylines(canvas, [quad], isClosed=True, color=(0,255,0), thickness=2)

    # 2) 좌석 ID 텍스트 (첫 번째 꼭짓점 기준 또는 중앙)
    cx = int(np.mean(quad[:,0])); cy = int(np.mean(quad[:,1]))  # 중심
    cv2.putText(canvas, str(sid), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

# ---------- 저장 및 확인 ----------
cv2.imwrite(OUT_PATH, canvas)
print(f'저장 완료 → {OUT_PATH}')

cv2.imshow('ROIs', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
