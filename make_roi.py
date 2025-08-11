import cv2, json
import numpy as np

IMG_PATH = 'subway2.jpg'
OUT_PATH = 'seat_roi2.json'
WIN_NAME = 'ROI Calibrator'

img      = cv2.imread(IMG_PATH)
canvas   = img.copy()

rois     = {}          # {id: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
seat_id  = 1
pts      = []          # 현재 그리고 있는 4점

def redraw():
    """canvas 갱신"""
    canvas[:] = img.copy()
    # 이미 저장된 ROI 그리기
    for sid, quad in rois.items():
        quad_np = np.array(quad, np.int32)
        cv2.polylines(canvas, [quad_np], True, (0,255,0), 1)
        # 첫 번째 꼭짓점에 ID 표기
        cv2.putText(canvas, str(sid), tuple(quad[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # 그리고 있는 ROI(pts) 그리기
    if pts:
        for p in pts:  # 점
            cv2.circle(canvas, p, 3, (0,0,255), -1)
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (0,0,255), 1)

def on_mouse(event, x, y, flags, param):
    global pts, seat_id
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        if len(pts) == 4:          # 4점 모두 찍은 경우
            rois[seat_id] = pts.copy()
            pts.clear()
            seat_id += 1
        redraw()

# ----- 메인 루프 -----
cv2.namedWindow(WIN_NAME)
cv2.setMouseCallback(WIN_NAME, on_mouse)
redraw()

print('[ENTER] 저장   [Z] 이전 점 삭제   [R] 전체 초기화   [ESC] 종료')
while True:
    cv2.imshow(WIN_NAME, canvas)
    key = cv2.waitKey(20) & 0xFF

    if key == 13:          # Enter → 저장
        with open(OUT_PATH, 'w') as f:
            json.dump(rois, f, indent=2)
        print('저장 완료 →', OUT_PATH)
        break

    elif key in (ord('z'), ord('Z')):   # 점 하나 되돌리기
        if pts:
            pts.pop()
            redraw()

    elif key in (ord('r'), ord('R')):   # 전체 초기화
        rois.clear(); pts.clear(); seat_id = 1; redraw()

    elif key == 27:        # Esc → 종료(저장 안 함)
        break

cv2.destroyAllWindows()
