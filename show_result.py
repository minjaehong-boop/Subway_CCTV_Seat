# seat_occupancy_from_fullframe.py
import cv2, json, numpy as np
from ultralytics import YOLO

IMG_PATH   = 'subway1.jpg'
MODEL_PATH = 'yolov8n-pose.pt'   # 포즈 모델 권장(엉덩이 키포인트 사용)

# ----- 좌석 ROI 로드 -----
rois = {k: np.array(v, np.int32) for k, v in json.load(open('seat_roi1.json')).items()}

def poly_centroid(poly: np.ndarray):
    M = cv2.moments(poly)
    if abs(M['m00']) < 1e-6:  # 면적 0 방지
        return tuple(poly.mean(axis=0).astype(int))
    cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
    return (cx, cy)

# 좌석 별 중심 미리 계산
seat_centers = {sid: poly_centroid(poly) for sid, poly in rois.items()}

# ----- 모델/이미지 로드 -----
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(IMG_PATH)
H, W = img.shape[:2]

model = YOLO(MODEL_PATH)
res   = model(img, conf=0.3, verbose=False)[0]

# ----- 사람 별 '앵커 포인트' 계산 -----
anchor_points = []  # [(x, y), ...]
boxes_xyxy    = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4))
use_pose      = (res.keypoints is not None) and (res.keypoints.xy is not None)

if use_pose:
    # COCO 포즈 인덱스: 11=left_hip, 12=right_hip (Ultralytics는 xy 텐서)
    kps = res.keypoints.xy.cpu().numpy()  # [N, 17, 2]
    for i, xyxy in enumerate(boxes_xyxy):
        if i >= len(kps):
            # 혹시 길이가 다르면 bbox 하단 중앙으로 fallback
            x1,y1,x2,y2 = xyxy
            anchor_points.append((int((x1+x2)/2), int(y2)))
            continue
        left_hip  = kps[i, 11]
        right_hip = kps[i, 12]
        # 힙 키포인트가 유효하면 중점, 아니면 bbox 하단 중앙
        if np.all(left_hip > 0) and np.all(right_hip > 0):
            ax = int((left_hip[0] + right_hip[0]) / 2)
            ay = int((left_hip[1] + right_hip[1]) / 2)
            anchor_points.append((ax, ay))
        else:
            x1,y1,x2,y2 = xyxy
            anchor_points.append((int((x1+x2)/2), int(y2)))
else:
    # 일반 사람탐지 모델일 때: 바운딩박스 하단 중앙
    for xyxy in boxes_xyxy:
        x1,y1,x2,y2 = xyxy
        anchor_points.append((int((x1+x2)/2), int(y2)))

# ----- 좌석 배정(1:1 매칭) -----
# 규칙:
# 1) 앵커 포인트가 좌석 폴리곤 내부인 좌석 후보를 모은다.
# 2) 후보가 여러 개면 좌석 중심과의 거리가 가장 가까운 좌석 선택.
# 3) 어떤 좌석도 포함하지 않으면, 좌석 중심과의 거리가 작은 것 중
#    일정 거리(th_pix) 이하이며 바운딩박스와 IoU>t_iou 인 좌석만 고려(옵션).
def point_in_poly(pt, poly):
    # >0: 내부, 0: 경계, <0: 외부
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def bbox_iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

assigned_seat = {}   # person_idx -> seat_id
taken_seats   = set()

# 1차: 내부 포함으로 매칭
for i, (ax, ay) in enumerate(anchor_points):
    candidates = [sid for sid, poly in rois.items() if point_in_poly((ax, ay), poly)]
    if len(candidates) == 1:
        if candidates[0] not in taken_seats:
            assigned_seat[i] = candidates[0]; taken_seats.add(candidates[0])
    elif len(candidates) > 1:
        # 가장 가까운 좌석 중심 선택
        best = None; best_d = 1e9
        for sid in candidates:
            cx, cy = seat_centers[sid]
            d = (ax-cx)**2 + (ay-cy)**2
            if sid not in taken_seats and d < best_d:
                best_d = d; best = sid
        if best is not None:
            assigned_seat[i] = best; taken_seats.add(best)

# 2차: 내부 미포함인 사람에 대해 근접+IoU 기준으로 보완 매칭(옵션)
th_pix = max(H, W) * 0.06   # 프레임 크기 기반 거리 임계(경험값)
t_iou  = 0.02               # 아주 작은 겹침만 있어도 인정
for i, (ax, ay) in enumerate(anchor_points):
    if i in assigned_seat:
        continue
    # 가장 가까운 좌석 찾기
    order = sorted(rois.keys(), key=lambda sid: (ax-seat_centers[sid][0])**2 + (ay-seat_centers[sid][1])**2)
    for sid in order:
        if sid in taken_seats: 
            continue
        cx, cy = seat_centers[sid]
        d = ((ax-cx)**2 + (ay-cy)**2) ** 0.5
        if d > th_pix:
            continue
        # bbox-좌석 사각 경계 IoU(좌석 폴리곤의 외접 사각형으로 근사)
        x1,y1,x2,y2 = boxes_xyxy[i]
        rx, ry, rw, rh = cv2.boundingRect(rois[sid])
        iou = bbox_iou_xyxy([x1,y1,x2,y2], [rx,ry,rx+rw,ry+rh])
        if iou >= t_iou:
            assigned_seat[i] = sid; taken_seats.add(sid); break

# ----- 결과 집계 & 시각화 -----
occupied = {sid: False for sid in rois.keys()}
for i, sid in assigned_seat.items():
    occupied[sid] = True

vis = img.copy()

# 좌석 폴리곤 및 상태 그리기
for sid, poly in rois.items():
    color = (0, 0, 255) if occupied[sid] else (0, 255, 0)
    cv2.polylines(vis, [poly], True, color, 2, cv2.LINE_AA)
    cx, cy = seat_centers[sid]
    cv2.putText(vis, f'{sid} {"Occ" if occupied[sid] else "Emp"}', (cx-20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# 사람 앵커 포인트/박스 그리기
for i, (ax, ay) in enumerate(anchor_points):
    cv2.circle(vis, (ax, ay), 4, (255, 255, 255), -1, cv2.LINE_AA)
    x1,y1,x2,y2 = map(int, boxes_xyxy[i])
    cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 1, cv2.LINE_AA)
    tag = assigned_seat.get(i, 'NA')
    cv2.putText(vis, f'P{i}->{tag}', (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

cv2.imwrite('seat_occupancy_result.jpg', vis)
for sid in sorted(rois.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
    print(f'Seat {sid}:', 'Occupied' if occupied[sid] else 'Empty')

print('완료: seat_occupancy_result.jpg 저장')
