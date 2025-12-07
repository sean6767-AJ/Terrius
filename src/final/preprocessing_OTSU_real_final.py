import cv2
import numpy as np

def center_image(img):
    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] < 5:
        return img

    cy, cx = np.mean(coords, axis=0)

    shift_x = int(np.round(14 - cx))
    shift_y = int(np.round(14 - cy))

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_NEAREST)
    return shifted


def preprocessing(frame):

    # 1) Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Blur   
    blur = gray

    H, W = blur.shape  # 또는 frame.shape[:2]

    y1 = int(H * 0.15)
    y2 = int(H * 0.90)
    x1 = int(W * 0.20)
    x2 = int(W * 0.80)

    roi = gray[y1:y2, x1:x2]

    # 3) OTSU threshold
    _, thresh = cv2.threshold(
       roi, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4) Morphology 
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # CCL로 숫자 blob 선택 (면적이 너무 큰 건 배경으로 제외)
    # 5) 숫자 하얀 픽셀만 기준으로 crop (CCL 대신)
    ys, xs = np.where(clean > 0)

    if len(xs) < 10:
        return np.zeros((28, 28), dtype=np.float32)

    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    digit_crop = clean[min_y:max_y+1, min_x:max_x+1]

    kernel = np.ones((2,2), np.uint8)
    
    digit_crop = cv2.morphologyEx(digit_crop, cv2.MORPH_OPEN, kernel)

    # 6) Resize longest side = 20 (MNIST 규격)
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

    # 7) 28×28 canvas 중앙 정렬
    canvas = np.zeros((28,28), dtype=np.uint8)
    x_pad = (28 - new_cols) // 2
    y_pad = (28 - new_rows) // 2

    canvas[y_pad:y_pad+new_rows, x_pad:x_pad+new_cols] = resized

    # 8) 숫자 중심 재정렬
    canvas = center_image(canvas)

    # 9) Normalize → MNIST 스타일 (숫자=1, 배경=0)
    digit_norm = canvas.astype(np.float32) / 255.0
    digit_norm = 1.0 - digit_norm

    return digit_norm