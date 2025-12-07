import cv2
import numpy as np

def find_board_roi(gray):
    """
    검정 종이(큰 사각형)를 contour 로 찾아 ROI 영역 반환
    """
    # 블러 + otsu
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 큰 contour 찾기
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  # 실패

    # 가장 큰 contour = 검정 네모판으로 가정
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    # ROI 부분만 반환
    return (x, y, w, h)


def preprocessing_with_board(frame):
    # 1) Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) 먼저 검정판 찾기
    roi_box = find_board_roi(gray)
    if roi_box is None:
        # 실패하면 빈 이미지 반환
        return np.zeros((28,28), dtype=np.float32)

    x, y, w, h = roi_box

    # 3) 검정판 영역 잘라오기
    board = gray[y:y+h, x:x+w]

    # -------- 여기서부터는 기존 숫자 blob 전처리 ---------

    # blur
    blur = cv2.GaussianBlur(board, (5,5), 0)

    # threshold
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # CCL (숫자 blob 잡기)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)

    if num_labels <= 1:
        return np.zeros((28,28), dtype=np.float32)

    # 면적 기준 숫자 선택
    areas = stats[:, cv2.CC_STAT_AREA]
    best_label = np.argmax(areas[1:]) + 1

    x2 = stats[best_label, cv2.CC_STAT_LEFT]
    y2 = stats[best_label, cv2.CC_STAT_TOP]
    w2 = stats[best_label, cv2.CC_STAT_WIDTH]
    h2 = stats[best_label, cv2.CC_STAT_HEIGHT]

    digit_crop = thresh[y2:y2+h2, x2:x2+w2]

    # resize longest side = 20
    rows, cols = digit_crop.shape
    if rows > cols:
        factor = 20 / rows
        new_rows, new_cols = 20, int(cols * factor)
    else:
        factor = 20 / cols
        new_cols, new_rows = 20, int(rows * factor)

    resized = cv2.resize(digit_crop, (new_cols, new_rows), interpolation=cv2.INTER_NEAREST)

    # canvas 28x28
    canvas = np.zeros((28,28), dtype=np.uint8)
    x_pad = (28 - new_cols) // 2
    y_pad = (28 - new_rows) // 2
    canvas[y_pad:y_pad+new_rows, x_pad:x_pad+new_cols] = resized

    # normalize MNIST style (숫자=1, 배경=0)
    digit_norm = canvas.astype(np.float32) / 255.0
    digit_norm = 1.0 - digit_norm

    return digit_norm
