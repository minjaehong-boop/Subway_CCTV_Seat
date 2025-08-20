#!/usr/bin/env python3
# detect2roi.py
# - CPU 전용
# - N프레임마다 처리(--frame-step)
# - "상태가 1초 이상 유지"될 때만 JSON 출력(디바운스)
#python detect2roi.py --video test_video2.mp4 --roi seat_roi.json --frame-step 10

import cv2, json, numpy as np, argparse, sys
from ultralytics import YOLO

HOLD_SEC = 1.0  # 1초 유지

def poly_centroid(poly: np.ndarray):
    M = cv2.moments(poly)
    if abs(M['m00']) < 1e-6:
        return tuple(poly.mean(axis=0).astype(int))
    cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def point_in_poly(pt, poly):
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

def load_and_scale_rois(roi_path, target_w, target_h):
    data = json.load(open(roi_path, "r", encoding="utf-8"))
    if isinstance(data, dict) and "meta" in data and "rois" in data:
        ref_w = data["meta"].get("ref_w", target_w)
        ref_h = data["meta"].get("ref_h", target_h)
        raw = data["rois"]
    else:
        ref_w, ref_h = target_w, target_h
        raw = data
    sx = target_w / float(ref_w); sy = target_h / float(ref_h)
    rois = {sid: (np.array(pts, np.float32) * [sx, sy]).astype(np.int32)
            for sid, pts in raw.items()}
    return rois

def compute_occupancy(img, model, rois, seat_centers, conf=0.3):
    res = model(img, conf=conf, verbose=False, device='cpu')[0]  # CPU 고정
    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4))
    use_pose   = (res.keypoints is not None) and (res.keypoints.xy is not None)

    anchor_points = []
    if use_pose:
        kps = res.keypoints.xy.cpu().numpy()
        for i, xyxy in enumerate(boxes_xyxy):
            if i >= len(kps):
                x1,y1,x2,y2 = xyxy
                anchor_points.append((int((x1+x2)/2), int(y2))); continue
            left_hip  = kps[i, 11]; right_hip = kps[i, 12]
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

    assigned_seat, taken_seats = {}, set()

    # 1차: 포인트 ∈ 폴리곤
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

    # 2차: 근접 + IoU
    H, W = img.shape[:2]
    th_pix = max(H, W) * 0.06; t_iou = 0.02
    for i, (ax, ay) in enumerate(anchor_points):
        if i in assigned_seat: continue
        order = sorted(rois.keys(), key=lambda sid: (ax-seat_centers[sid][0])**2 + (ay-seat_centers[sid][1])**2)
        for sid in order:
            if sid in taken_seats: continue
            cx, cy = seat_centers[sid]
            d = ((ax-cx)**2 + (ay-cy)**2) ** 0.5
            if d > th_pix: continue
            x1,y1,x2,y2 = boxes_xyxy[i]
            rx, ry, rw, rh = cv2.boundingRect(rois[sid])
            iou = bbox_iou_xyxy([x1,y1,x2,y2], [rx,ry,rx+rw,ry+rh])
            if iou >= t_iou:
                assigned_seat[i] = sid; taken_seats.add(sid); break

    occupied = {sid: False for sid in rois.keys()}
    for _, sid in assigned_seat.items():
        occupied[sid] = True
    return occupied

def state_equal(a, b):
    if a is None or b is None: return False
    if len(a) != len(b): return False
    for k in a.keys():
        if a[k] != b.get(k): return False
    return True

def try_seek(cap, next_idx):
    return bool(cap.set(cv2.CAP_PROP_POS_FRAMES, int(next_idx)))

def main():
    ap = argparse.ArgumentParser(description="CPU 전용, N프레임마다 처리 + 1초 디바운스 출력")
    ap.add_argument("--video", default="test_video2.mp4", help="입력 영상 경로")
    ap.add_argument("--roi", default="seat_roi.json", help="ROI JSON 경로")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics 모델 경로")
    ap.add_argument("--conf", type=float, default=0.3, help="탐지 conf")
    ap.add_argument("--frame-step", type=int, default=5, help="N프레임마다 처리(>=1)")
    args = ap.parse_args()

    step = max(1, args.frame_step)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("영상을 열 수 없습니다."); sys.exit(1)

    # 첫 프레임
    ok, frame = cap.read()
    if not ok:
        print("첫 프레임을 읽을 수 없습니다."); sys.exit(1)

    H, W = frame.shape[:2]
    rois = load_and_scale_rois(args.roi, W, H)
    rois = {k: np.array(v, np.int32) for k, v in rois.items()}
    seat_centers = {sid: poly_centroid(poly) for sid, poly in rois.items()}

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else None

    model = YOLO(args.model)
    model.to('cpu')  # CPU 고정

    # 안정화 상태 머신 초기화(처음 상태도 1초 유지 후에만 출력)
    stable_state = None
    pending_state = None
    pending_since = None

    # 현재 프레임 인덱스
    fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    fno = 1 if fno <= 1 else fno

    # 처리 루프 (첫 프레임 포함해서 step 간격으로)
    while True:
        # 시간(초)
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            t = pos_msec / 1000.0
        else:
            t = (fno - 1) / fps if fps else 0.0

        # 좌석 점유 즉시추정
        occ = compute_occupancy(frame, model, rois, seat_centers, conf=args.conf)

        # 디바운스 로직: 같은 상태가 HOLD_SEC 동안 지속되어야만 출력
        if stable_state is None:
            # 아직 확정 상태가 없으면 후보 상태 관찰
            if pending_state is None or not state_equal(occ, pending_state):
                pending_state = occ.copy()
                pending_since = t
            else:
                if (t - pending_since) >= HOLD_SEC:
                    stable_state = pending_state.copy()
                    print(json.dumps({"frame": fno, "t": round(t,3), "seats": stable_state}, ensure_ascii=False))
        else:
            if state_equal(occ, stable_state):
                # 안정 상태 유지 → 후보 초기화
                pending_state = None
                pending_since = None
            else:
                if pending_state is None or not state_equal(occ, pending_state):
                    pending_state = occ.copy()
                    pending_since = t
                else:
                    if (t - pending_since) >= HOLD_SEC:
                        stable_state = pending_state.copy()
                        print(json.dumps({"frame": fno, "t": round(t,3), "seats": stable_state}, ensure_ascii=False))
                        pending_state = None
                        pending_since = None

        # 다음 처리 프레임으로 점프
        next_idx = fno + step
        jumped = try_seek(cap, next_idx)
        if jumped:
            ok, frame = cap.read()
            if not ok: break
            fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if fno <= 0: fno = next_idx
        else:
            left = step
            ok = True
            while left > 0 and ok:
                ok = cap.grab()
                left -= 1
            if not ok: break
            ok, frame = cap.retrieve()
            if not ok: break
            fno += step

    cap.release()

if __name__ == "__main__":
    main()
