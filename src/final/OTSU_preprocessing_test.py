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

    # 3) OTSU threshold
    _, thresh = cv2.threshold(
       blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4) Morphology 
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 5) CCL로 가장 큰 숫자 blob 선택
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)

    empty = np.zeros((28,28), dtype=np.uint8)
    if num_labels <= 1:
        return empty.astype(np.float32)

    areas = stats[:, cv2.CC_STAT_AREA]
    best_label = np.argmax(areas[1:]) + 1

    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    w = stats[best_label, cv2.CC_STAT_WIDTH]
    h = stats[best_label, cv2.CC_STAT_HEIGHT]

    digit_crop = clean[y:y+h, x:x+w]

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
