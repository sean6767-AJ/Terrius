import cv2
import numpy as np
from scipy import stats

def preprocessing(frame):

    # 1) BGR > GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2) blur(노이즈 제거) > 배열저장(이미지)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 3) sharp
    # 커널 생성(대상이 있는 픽셀을 강조) 커널을 이미지 위를 슬라이딩 하며 convolution 수행 > 숫자 윤곽 선명하게
    # 가운데 값 5 : 중심 픽셀 5배 강조, 주변 값 -1 : 주변픽셀들을 빼서 대비효과, 0 : 그 방향은 영향x
    # cv2.filter2D(blur, -1, kernel) > blur된 이미지에 kernel값 적용
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])   
    sharp = cv2.filter2D(blur, -1, kernel)
    
    # Threshold 
    # binary 이미지(흑 / 백) 생성
    # adaptive threshold > OTSU threshold 변경
    _, thresh = cv2.threshold(
        sharp, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # theshold에 morphology 적용
    kernel = np.ones((3,3), np.uint8)
    
    # opening : noise 제거
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # closing : 숫자 두껍게 & 끊어진 부분 이어줌
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # CCL(Connected Components With stats) 활용 > 숫자 덩어리(blob) 찾아서 crop
    # 숫자 blob은 사진의 중앙 근처에 존재 
    # blob 분석을 위한 정보
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)

    # 숫자 blob이 없을 때 빈 캔버스 리턴
    if num_labels <= 1:
        empty = np.zeros((28,28), dtype=np.uint8)
        return empty

    # 노이즈 제거용 면적 값
    areas = stats[:, cv2.CC_STAT_AREA]

    # 이미지(closed)의 크기 : h(height), w(width) 받아옴
    h, w = closed.shape
    # 프레임의 정중앙 좌표
    cx_frame = w // 2
    cy_frame = h // 2

    best_label = None # 선택된 blob은 label로 표시됨
    best_dist = 1e20  # 거리 비교용 값(loop돌면서 dist 갱신)

    for label in range(1, num_labels):
        
        # 작은 노이즈 제거
        if areas[label] < 50:
            continue 
        
        # blob의 bounding box 정보(왼쪽:x 위쪽:y 폭:bw 높이:bh)
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        # blob의 중심 좌표 계산
        cx = x+bw // 2
        cy = y+bh // 2

        # 프레임 중앙(cx,cy)과의 거리 계산
        # dist가 작을수록 중앙에 있음(숫자 blob일 확률 큼)
        dist = abs(cx_frame - cx) + abs(cy_frame - cy)

        # loop를 돌 때마다 점점 dist가 작은 값을 선택 > 중앙에 가까운 blob 선택(숫자)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    
    # best_label을 찾지 못했을 경우 > areas가 가장 넓은 label 선택([1:] > 0번(배경)라벨 빼고 계산 > 슬라이싱 했으므로 +1 해줘야 실제 label)
    if best_label is None:
        best_label = np.argmax(areas[1:]) + 1

    # crop 좌표
    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]

    # crop
    digit_crop = closed[y:y+bh, x:x+bw]

    # corp 후 MNIST input용으로 resize
    # MNIST의 숫자는 20x20, 배경은 28x28
    # 긴변을 20으로 resize, 나머지 변은 긴변을 기준으로 맞춤 > 28x28 canvas에 합성

    # crop의 사이즈 불러오기
    rows, cols = digit_crop.shape

    # 이상한 crop이 나오면 빈 캔버스
    if rows == 0 or cols == 0:
        return empty

    # 긴변 20으로 resize
    if rows > cols:
        factor = 20 / rows # 비율 계산용 상수
        final_rows = 20
        final_cols = int(cols * factor)
    else:
        factor = 20 / cols
        final_cols = 20
        final_rows = int(rows * factor)
    
    digit_resized = cv2.resize(digit_crop, (final_cols, final_rows))

    # 28x28 canvas 생성
    canvas = np.zeros((28,28), dtype=np.uint8)

    # 여백(pad) 계산 
    # 28x28 - (final_cols, final_rows) 값을 2로 나눠 왼쪽,오른쪽 위,아래 여백으로 설정 > canvas 중앙에 20x20 숫자(blob) 붙이기
    x_pad = (28 - final_cols) // 2
    y_pad = (28 - final_rows) // 2

    canvas[y_pad : y_pad + final_rows, x_pad : x_pad + final_cols] = digit_resized

    digit_norm = canvas / 255.0

    digit_norm = 1.0 - (canvas / 255.0)

    return digit_norm

