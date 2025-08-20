# Subway_CCTV_Seat
Seoul_Subway(Line_2)_CCTV_seat_detection


## 프로젝트 개요
- `make_roi.py`로 좌석별 다각형 ROI를 설정하고, 초기에는 ROI를 잘라서 사람을 검출했으나 제한된 cctv 시점에 의한 잘림과 누락 문제가 발생.
  (`roi_visualize.py`로 ROI를 확인한 결과, ROI → 크롭 → 탐지 방식은 앉은 사람 일부가 잘려서 Empty로 잘못 판정됨)
- 이를 개선하기 위해 `entire_frame.py`에서 전체 프레임에서 먼저 사람을 탐지하는 구조로 변경.
- `detect2roi_final.py`에서는 바운딩박스 대신 엉덩이(hip) 키포인트 중점이 좌석 바운딩 박스 내부에 포함되는지로 점유 여부를 판단.
- 결과적으로 겹침·경계 근처에서도 안정적인 좌석 점유 판정이 가능해졌으며, 1:1 매칭과 근접+IoU 보완 로직을 통해 오탐을 줄임.
detect2roi.py---최종 10프레임마다 추론하는 빠른 버전
detect2roi2.py---영상이랑 같이 확인하는 버전
make_roi_final.py---영상에서 roi를 만드는 코드
모든 과정은 test_video2.mp4를 사용하면됨.
코드 실행 커맨드는 맨 위 주석에 담겨있음
---

## detect2roi_final.py 핵심 코드 블럭 분석

### 1. `poly_centroid(poly)`
```python
def poly_centroid(poly: np.ndarray):
    M = cv2.moments(poly)
    if abs(M['m00']) < 1e-6:
        return tuple(poly.mean(axis=0).astype(int))
    cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
    return (cx, cy)
````

* 좌석 ROI 다각형의 중심 좌표 계산.
* 한 사람이 여러 좌석에 걸친 경우, 중심과의 거리를 비교해 가장 가까운 좌석을 선택하기 위해 사용.

### 2. `point_in_poly(pt, poly)`

```python
def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0
```

* 한 점이 좌석 폴리곤 내부 또는 경계에 있는지 판정.
* 1차 매칭에서 힙 중점이 해당 좌석 안에 있는지를 확인하는 핵심 함수.

### 3. `bbox_iou_xyxy(a, b)`

```python
def bbox_iou_xyxy(a, b):
    ...
```

* 사람 바운딩박스와 좌석 외접사각형 간의 IoU 계산.
* 2차 보완 매칭에서 “가까움 + 일정 겹침” 조건을 만족하는지 판단.

### 4. `compute_occupancy(img, model, conf=0.3)`

* **사람 탐지 & 앵커 포인트 계산**: YOLO pose 모델로 탐지 후, 힙(11,12) 좌표의 중점을 앵커 포인트로 사용. 힙 좌표가 없으면 바운딩박스 하단 중앙을 사용(필요시 위치 수정 가능)
* **1차 매칭**: 앵커 포인트가 좌석 ROI 안에 들어가면 그 좌석에 배정. 여러 후보 시 좌석 중심과의 거리로 결정.
* **2차 보완**: 1차에서 누락된 사람에 대해, 좌석 중심과의 거리 임계값 + IoU 기준을 만족하면 해당 좌석으로 배정.
* **출력**: 최종적으로 좌석별 점유 여부 딕셔너리(`occupied`)를 반환.(앱으로의 입력 형식에 맞게 수정 가능)

---

## 점유 여부 판단 흐름 (예: 11번 좌석)

1. 전체 프레임에서 사람을 탐지하고, 힙 위치(또는 하단 중앙점)를 계산.
2. 이 점이 11번 좌석 ROI 안에 있으면 바로 Occupied로 기록.
3. ROI 안에 없더라도, 11번 좌석 중심에 충분히 가깝고 박스가 11번 외접사각형과 조금이라도 겹치면 Occupied로 기록.
4. 최종적으로 11번 좌석이 한 명이라도 배정되면 `Occupied`, 아니면 `Empty`.

---

## 파일별 기능

* **make\_roi.py**: 마우스 클릭으로 좌석 다각형 ROI 생성 후 JSON 저장.
* **roi\_visualize.py**: ROI 시각화 및 ID 표시.
* **roi2detect.py**: ROI 안에서만 사람 탐지 후 점유 여부 출력(초기 방식).
* **entire\_frame.py**: 전체 프레임에서 사람 탐지 후 바운딩박스 출력.
* **detect2roi\_final.py**: 전체 프레임 사람 탐지 → 엉덩이 위치 기준 ROI 매칭 → 좌석 점유 여부 출력(최종 방식)-좌석 점유 결과를 JSON으로 표준 출력

---


## 실행 예시 (JSON 출력)

### 1) 이미지 파일

```bash
python detect2roi_final.py subway.jpg
```

출력 예:

```json
{"1": false, "2": false, "3": true, "4": false, "5": true}
```

> 키는 좌석 ID, 값은 `true`(Occupied) / `false`(Empty)

### 2) 영상 파일

```bash
python detect2roi_final.py subway.mp4
```

출력 예(**매 프레임마다 한 줄씩**):

```json
{"frame": 1, "seats": {"1": false, "2": false, "3": true}}
{"frame": 2, "seats": {"1": false, "2": false, "3": true}}
{"frame": 3, "seats": {"1": false, "2": true,  "3": true}}
...
```

### 3) 웹캠

```bash
python detect2roi_final.py 0
```

출력 예(**매 프레임마다 한 줄씩**):

```json
{"frame": 17, "seats": {"11": true, "12": false, "13": false}}
{"frame": 18, "seats": {"11": true, "12": true,  "13": false}}
...
```

---

> 앱/서버에서 바로 파싱 가능하도록 표준 JSON을 사용함. 필요 시 키 이름(예: `seats` → `data`) 등은 쉽게 변경 가능.
> 
> 주의. 영상/웹캠은 **프레임마다 출력**하므로, 소비 측(앱/서버)에서 필요 시 \*\*변화 감지(이전 상태와 비교)\*\*를 적용해 트래픽을 줄이세요.
