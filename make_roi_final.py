#!/usr/bin/env python3
# make_roi.py : video 프레임에서 좌석 ROI를 다각형으로 지정하여 JSON 저장(meta: ref_w/ref_h 포함)
import cv2, json, numpy as np
import argparse
from pathlib import Path

def draw_help(canvas):
    hts = [
        "[H] 도움말 토글",
        "[F/B] 다음/이전 프레임",
        "[Enter] 현재 프레임 고정 후 ROI 편집 시작",
        "[Left Click] 꼭짓점 추가, [Enter] 폴리곤 닫기",
        "[U] 마지막 점 취소, [D] 마지막 ROI 삭제",
        "[S] 저장, [Q] 종료"
    ]
    y = 24
    for t in hts:
        cv2.putText(canvas, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        y += 24

def main():
    ap = argparse.ArgumentParser(description="영상에서 좌석 ROI 작성기")
    ap.add_argument("--video", required=True, help="입력 영상 경로 (예: test_video.mp4)")
    ap.add_argument("--out", default="seat_roi.json", help="ROI 저장 JSON 경로")
    ap.add_argument("--start", type=int, default=0, help="시작 프레임 인덱스")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, min(args.start, total-1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("첫 프레임을 읽을 수 없습니다.")
    H, W = frame.shape[:2]

    window = "make_roi(video)"
    cv2.namedWindow(window)
    show_help = True
    roi_mode = False      # False: 프레임 선택 모드, True: ROI 편집 모드
    rois = []             # list of np.array([[x,y],...], int32)
    current = []          # 현재 그리고 있는 폴리곤 점 목록
    chosen_frame_idx = idx

    def on_mouse(event, x, y, flags, param):
        nonlocal current
        if not roi_mode:  # 프레임 선택 모드에서는 클릭 무시
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])

    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        if show_help:
            draw_help(canvas)

        if roi_mode:
            # 그려진 ROI
            for i, poly in enumerate(rois, start=1):
                cv2.polylines(canvas, [poly], True, (0,255,0), 2, cv2.LINE_AA)
                cx, cy = int(poly[:,0].mean()), int(poly[:,1].mean())
                cv2.putText(canvas, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            # 현재 점
            if len(current) > 0:
                pts = np.array(current, np.int32)
                cv2.polylines(canvas, [pts], False, (255,255,255), 2, cv2.LINE_AA)
                for p in current:
                    cv2.circle(canvas, tuple(p), 4, (255,255,255), -1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"Frame {idx}/{max(total-1,0)}", (10, canvas.shape[0]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)

        cv2.imshow(window, canvas)
        k = cv2.waitKey(20) & 0xFF

        if k in (ord('h'), ord('H')):
            show_help = not show_help
        elif k in (ord('f'), ord('F')) and not roi_mode:
            idx = min(idx+1, total-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok: break
        elif k in (ord('b'), ord('B')) and not roi_mode:
            idx = max(idx-1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok: break
        elif k == 13:  # Enter
            if not roi_mode:
                # 이 프레임을 기준 프레임으로 고정하고 ROI 편집 시작
                roi_mode = True
                chosen_frame_idx = idx
            else:
                # ROI 편집 중이면, 폴리곤 닫기
                if len(current) >= 3:
                    rois.append(np.array(current, np.int32))
                    current = []
        elif k in (ord('u'), ord('U'), 8):  # Backspace
            if roi_mode:
                if current: current.pop()
        elif k in (ord('d'), ord('D')):
            if roi_mode and rois:
                rois.pop()
        elif k in (ord('s'), ord('S')):
            if not rois:
                print("ROI가 없습니다. 저장 불가.")
                continue
            rois_dict = {}
            for i, poly in enumerate(rois, start=1):
                rois_dict[str(i)] = poly.astype(int).tolist()
            out = {
                "meta": {"ref_w": W, "ref_h": H, "source": "video", "frame": int(chosen_frame_idx)},
                "rois": rois_dict
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"저장 완료 → {args.out}")
            break
        elif k in (ord('q'), ord('Q'), 27):  # ESC
            print("저장하지 않고 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
