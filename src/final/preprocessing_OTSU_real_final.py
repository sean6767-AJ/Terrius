import cv2
import numpy as np


# 20x20 숫자 픽셀의 좌표를 기준으로 숫자의 무게 중심을 계산해 이 중심이 28x28 빈 canvas의 정중앙에 오도록 정렬
def center_image(img):
    # img > 0 인 모든 픽셀 좌표(숫자 픽셀)를 수집
    # coords는 (y, x) 형태의 좌표 리스트
    coords = np.column_stack(np.where(img > 0))

    # 숫자 픽셀이 거의 없으면(노이즈 등) 이동할 의미가 없으므로 그대로 반환
    if coords.shape[0] < 5:
        return img

    # 숫자 픽셀들의 평균 좌표 > 무게중심
    # cy: 세로 방향 중심, cx: 가로 방향 중심
    cy, cx = np.mean(coords, axis=0)

    # 28×28 이미지의 정중앙은 (14, 14) 이므로,
    # 현재 무게중심을 중앙으로 이동시키기 위한 shift 거리 계산
    shift_x = int(np.round(14 - cx))  # 좌우 이동량
    shift_y = int(np.round(14 - cy))  # 상하 이동량

    # 평행 이동 변환 행렬 생성
    # [[1, 0, shift_x],
    #  [0, 1, shift_y]]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # warpAffine: 이미지를 M 행렬대로 이동(shift)
    # 숫자의 무게중심이 정확히 중앙(14,14)에 오도록 정렬됨
    shifted = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_NEAREST)

    return shifted


def preprocessing(frame):

    # Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur (가우시안 실패)
    blur = gray

    H, W = blur.shape  

    # ROI
    # 이미지 내부 숫자만을 검출하기 위한 사각형 생성
    y1 = int(H * 0.15)
    y2 = int(H * 0.90)
    x1 = int(W * 0.20)
    x2 = int(W * 0.80)

    roi = gray[y1:y2, x1:x2]

    #  OTSU threshold
    # ROI : 전처리 된 gray 의 숫자 부분
    # THRESH_BINARY_INV : 숫자를 흰색, 배경을 검정색으로 반전
    # THRESH_OTSU : 최적 임계값을 자동 계산하여 이진화
    _, thresh = cv2.threshold(
       roi, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
   
    #  Morphology 
    kernel = np.ones((3,3), np.uint8)
     
    # opening : noise 제거
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # CCL로 숫자 blob 선택 (면적이 너무 큰 건 배경으로 제외) > 실패
    # 5) 숫자 하얀 픽셀만 기준으로 crop 

    # 흰색 픽셀들 좌표 반환
    ys, xs = np.where(clean > 0) 

    # 흰색 픽셀이 10개 미만이면 숫자가 없는 것으로 판단 (CNN 오류 방지)
    if len(xs) < 10:
        return np.zeros((28, 28), dtype=np.float32)

    # 숫자를 감싸는 bounding box 계산
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    # bounding box를 이용해 숫자 crop
    digit_crop = clean[min_y:max_y+1, min_x:max_x+1]

    kernel = np.ones((2,2), np.uint8)
    
    digit_crop = cv2.morphologyEx(digit_crop, cv2.MORPH_OPEN, kernel)

    # 긴변 20으로 resize (MNIST 규격)
    rows, cols = digit_crop.shape

    if rows > cols:
        factor = 20 / rows
        new_rows, new_cols = 20, int(cols * factor)
    else:
        factor = 20 / cols
        new_cols, new_rows = 20, int(rows * factor)

    resized = cv2.resize(
        digit_crop, (new_cols, new_rows),
        interpolation=cv2.INTER_NEAREST     
    )

    # 28×28 canvas 중앙 정렬
    canvas = np.zeros((28,28), dtype=np.uint8)
    x_pad = (28 - new_cols) // 2
    y_pad = (28 - new_rows) // 2

    canvas[y_pad:y_pad+new_rows, x_pad:x_pad+new_cols] = resized

    # 숫자 중심 재정렬
    canvas = center_image(canvas)

    # Normalize (숫자 : 흰색(1), 배경 : (0))
    digit_norm = canvas.astype(np.float32) / 255.0
    digit_norm = 1.0 - digit_norm

    return digit_norm   