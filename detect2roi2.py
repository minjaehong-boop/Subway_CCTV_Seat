#!/usr/bin/env python3
# detect2roi2.py : 프레임별 즉시추정 vs 1.5초 유지 안정화 결과를 좌우로 보여주고,
# 안정화 상태가 바뀔 때만 JSON을 출력하는 스크립트
#python detect2roi2.py test_video.mp4 --roi seat_roi.json --hold 1.5 --show

import cv2, json, numpy as np, argparse, sys
from ultralytics import YOLO

# ---------- 기본 유틸 ----------
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

# ---------- 좌석 점유 계산(기존 로직) ----------
def compute_occupancy(img, model, rois, seat_centers, conf=0.3):
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

# ---------- 시각화(좌우 비교) ----------
def draw_overlay(base, rois, seat_centers, occ, title=None, sub=None):
    vis = base.copy()
    for sid, poly in rois.items():
        color = (0, 0, 255) if occ.get(sid, False) else (0, 200, 0)  # red=Occ, green=Emp
        cv2.polylines(vis, [poly], True, color, 2, cv2.LINE_AA)
        cx, cy = seat_centers[sid]
        cv2.putText(vis, f"{sid}:{'O' if occ.get(sid, False) else 'E'}",
                    (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if title:
        cv2.putText(vis, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(vis, sub, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
    return vis

def state_equal(a, b):
    if a is None or b is None: return False
    if len(a) != len(b): return False
    for k in a.keys():
        if a[k] != b.get(k): return False
    return True

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser(description="좌석 점유(1.5초 디바운스) + 좌우 비교 뷰 + 안정화 변경만 JSON 출력")
    ap.add_argument("video", help="입력 영상 경로 (예: test_video.mp4)")
    ap.add_argument("--roi", default="seat_roi.json", help="ROI JSON 경로")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics 모델 경로")
    ap.add_argument("--conf", type=float, default=0.3, help="탐지 conf")
    ap.add_argument("--hold", type=float, default=1.5, help="안정화 대기 시간(초)")
    ap.add_argument("--show", action="store_true", help="영상 창 표시(좌우 비교)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("영상을 열 수 없습니다."); sys.exit(1)

    # 첫 프레임/ROI 스케일
    ok, frame = cap.read()
    if not ok:
        print("첫 프레임을 읽을 수 없습니다."); sys.exit(1)

    H, W = frame.shape[:2]
    rois = load_and_scale_rois(args.roi, W, H)
    rois = {k: np.array(v, np.int32) for k, v in rois.items()}
    seat_centers = {sid: poly_centroid(poly) for sid, poly in rois.items()}

    model = YOLO(args.model)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = None  # POS_MSEC로 대체
    fno = 1

    # 안정화 상태 머신
    stable_state = None         # 확정 상태(출력용)
    stable_since = 0.0          # 확정 상태가 시작된 시각(초)
    pending_state = None        # 후보 상태
    pending_since = 0.0         # 후보 상태가 시작된 시각(초)
    HOLD = float(args.hold)

    # 초기 즉시추정
    instant = compute_occupancy(frame, model, rois, seat_centers, conf=args.conf)
    # 초기 상태를 곧바로 확정/출력
    # 현재 시각
    t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
    stable_state = instant.copy()
    stable_since = t
    print(json.dumps({"frame": fno, "t": round(t,3), "seats": stable_state}, ensure_ascii=False))

    if args.show:
        left  = draw_overlay(frame, rois, seat_centers, instant, "Instant (per-frame)")
        right = draw_overlay(frame, rois, seat_centers, stable_state,
                             f"Stable (hold {HOLD:.1f}s)", "✓ current output")
        canvas = cv2.hconcat([left, right])
        cv2.imshow("Seat (Instant vs Stable)", canvas)

    # 본 루프
    while True:
        ok, frame = cap.read()
        if not ok: break
        fno += 1
        t = (cap.get(cv2.CAP_PROP_POS_MSEC) or (fno / fps if fps else 0.0)) / 1000.0

        # 1) 즉시추정
        instant = compute_occupancy(frame, model, rois, seat_centers, conf=args.conf)

        # 2) 안정화 로직(디바운스: HOLD초 유지 시에만 확정 상태 변경)
        if state_equal(instant, stable_state):
            # 확정 상태와 같으면 보류 상태 초기화
            pending_state = None
            pending_since = 0.0
        else:
            if pending_state is None or not state_equal(instant, pending_state):
                # 새로운 후보 상태 시작
                pending_state = instant.copy()
                pending_since = t
            else:
                # 같은 후보 상태가 계속 유지되는 중
                if (t - pending_since) >= HOLD:
                    # 확정 상태로 승격 + 출력
                    stable_state = pending_state.copy()
                    stable_since = t
                    print(json.dumps({"frame": fno, "t": round(t,3), "seats": stable_state}, ensure_ascii=False))
                    pending_state = None
                    pending_since = 0.0

        # 3) 시각화(좌우 비교)
        if args.show:
            # 왼쪽: 즉시추정
            sub_left = None
            # 오른쪽: 안정화 + (진행 바)
            sub_right = "✓ current output"
            if pending_state is not None:
                remain = max(0.0, HOLD - (t - pending_since))
                sub_right = f"pending... {remain:.1f}s to switch"

            left  = draw_overlay(frame, rois, seat_centers, instant, "Instant (per-frame)", sub_left)
            right = draw_overlay(frame, rois, seat_centers, stable_state, f"Stable (hold {HOLD:.1f}s)", sub_right)

            # 진행 바(오른쪽 상단)
            vis = right
            bar_w, bar_h = 200, 12
            x0, y0 = 10, 70
            cv2.rectangle(vis, (x0, y0), (x0+bar_w, y0+bar_h), (80,80,80), 1, cv2.LINE_AA)
            if pending_state is not None:
                ratio = np.clip((t - pending_since) / HOLD, 0.0, 1.0)
                cv2.rectangle(vis, (x0+1, y0+1), (x0+1+int((bar_w-2)*ratio), y0+bar_h-1), (0,200,255), -1, cv2.LINE_AA)

            canvas = cv2.hconcat([left, vis])
            cv2.imshow("Seat (Instant vs Stable)", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
