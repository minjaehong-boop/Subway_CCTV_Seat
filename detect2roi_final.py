# detect2roi_json.py
import cv2, json, numpy as np, os, sys
from ultralytics import YOLO

MODEL_PATH = 'yolov8n-pose.pt'
ROI_JSON   = 'seat_roi.json'

# ----- 좌석 ROI 로드 -----
rois = {k: np.array(v, np.int32) for k, v in json.load(open(ROI_JSON)).items()}

def poly_centroid(poly: np.ndarray):
    M = cv2.moments(poly)
    if abs(M['m00']) < 1e-6:
        return tuple(poly.mean(axis=0).astype(int))
    cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
    return (cx, cy)

seat_centers = {sid: poly_centroid(poly) for sid, poly in rois.items()}

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def bbox_iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def compute_occupancy(img, model, conf=0.3):
    H, W = img.shape[:2]
    res = model(img, conf=conf, verbose=False)[0]

    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4))
    use_pose   = (res.keypoints is not None) and (res.keypoints.xy is not None)

    anchor_points = []
    if use_pose:
        kps = res.keypoints.xy.cpu().numpy()  # [N,17,2]
        for i, xyxy in enumerate(boxes_xyxy):
            if i >= len(kps):
                x1,y1,x2,y2 = xyxy
                anchor_points.append((int((x1+x2)/2), int(y2)))
                continue
            left_hip  = kps[i, 11]
            right_hip = kps[i, 12]
            if np.all(left_hip > 0) and np.all(right_hip > 0):
                ax = int((left_hip[0] + right_hip[0]) / 2)
                ay = int((left_hip[1] + right_hip[1]) / 2)
                anchor_points.append((ax, ay))
            else:
                x1,y1,x2,y2 = xyxy
                anchor_points.append((int((x1+x2)/2), int(y2)))
    else:
        for xyxy in boxes_xyxy:
            x1,y1,x2,y2 = xyxy
            anchor_points.append((int((x1+x2)/2), int(y2)))

    assigned_seat = {}
    taken_seats   = set()

    # 1차 매칭
    for i, (ax, ay) in enumerate(anchor_points):
        candidates = [sid for sid, poly in rois.items() if point_in_poly((ax, ay), poly)]
        if len(candidates) == 1:
            if candidates[0] not in taken_seats:
                assigned_seat[i] = candidates[0]; taken_seats.add(candidates[0])
        elif len(candidates) > 1:
            best = None; best_d = 1e9
            for sid in candidates:
                cx, cy = seat_centers[sid]
                d = (ax-cx)**2 + (ay-cy)**2
                if sid not in taken_seats and d < best_d:
                    best_d = d; best = sid
            if best is not None:
                assigned_seat[i] = best; taken_seats.add(best)

    # 2차 보완
    th_pix = max(H, W) * 0.06
    t_iou  = 0.02
    for i, (ax, ay) in enumerate(anchor_points):
        if i in assigned_seat:
            continue
        order = sorted(rois.keys(), key=lambda sid: (ax-seat_centers[sid][0])**2 + (ay-seat_centers[sid][1])**2)
        for sid in order:
            if sid in taken_seats:
                continue
            cx, cy = seat_centers[sid]
            d = ((ax-cx)**2 + (ay-cy)**2) ** 0.5
            if d > th_pix:
                continue
            x1,y1,x2,y2 = boxes_xyxy[i]
            rx, ry, rw, rh = cv2.boundingRect(rois[sid])
            iou = bbox_iou_xyxy([x1,y1,x2,y2], [rx,ry,rx+rw,ry+rh])
            if iou >= t_iou:
                assigned_seat[i] = sid; taken_seats.add(sid); break

    occupied = {sid: False for sid in rois.keys()}
    for _, sid in assigned_seat.items():
        occupied[sid] = True
    return occupied

def is_image(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']

def main():
    if len(sys.argv) < 2:
        print('사용법. python seat_occupancy_core.py <이미지|영상 경로 또는 0>')
        sys.exit(1)

    src = sys.argv[1]
    model = YOLO(MODEL_PATH)

    # 웹캠
    if src == '0':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('웹캠을 열 수 없습니다.')
            sys.exit(1)
        fno = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fno += 1
            occ = compute_occupancy(frame, model)
            print(json.dumps({"frame": fno, "seats": occ}, ensure_ascii=False))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        return

    # 파일 입력
    if is_image(src):
        img = cv2.imread(src)
        if img is None:
            print('이미지를 열 수 없습니다.')
            sys.exit(1)
        occ = compute_occupancy(img, model)
        print(json.dumps(occ, ensure_ascii=False))
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print('영상을 열 수 없습니다.')
            sys.exit(1)
        fno = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fno += 1
            occ = compute_occupancy(frame, model)
            print(json.dumps({"frame": fno, "seats": occ}, ensure_ascii=False))
        cap.release()

if __name__ == '__main__':
    main()
