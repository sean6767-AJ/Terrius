import cv2
import numpy as np
from scipy import stats

def preprocess_debug(frame):
    results = {}  # 중간 결과 저장용 dict

    # 1) GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results["gray"] = gray

    # 2) BLUR
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    results["blur"] = blur

    # 3) SHARP
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(blur, -1, kernel)
    results["sharp"] = sharp

    # 4) HIST EQUALIZATION
    #equalized = cv2.equalizeHist(sharp)
    #results["equalization"] = equalized

    # 5) THRESH

    equalized = cv2.equalizeHist(sharp)
    results["equalization"] = equalized


    _, thresh = cv2.threshold(
        equalized, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    results["thresh"] = thresh

    # 6) CCL
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)
    results["labels"] = labels

    if num_labels <= 1:
        results["component_mask"] = np.zeros_like(thresh)
        results["digit"] = np.zeros_like(thresh)
        results["crop"] = np.zeros((28,28), dtype=np.uint8)
        results["canvas"] = np.zeros((28,28), dtype=np.uint8)
        return results

    areas = stats[:, cv2.CC_STAT_AREA]

    h, w = thresh.shape
    cx_frame = w // 2
    cy_frame = h // 2

    best_label = None
    best_dist = 1e20

    for label in range(1, num_labels):  # 배경(label=0) 제외
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        cx = x + bw // 2
        cy = y + bh // 2

    # 프레임 중앙에서 얼마나 떨어져 있는지 계산 (맨해튼 거리)
        dist = abs(cx - cx_frame) + abs(cy - cy_frame)

        # 너무 작은 blob은 무시 (노이즈 제거)
        if areas[label] < 50:
            continue

        if dist < best_dist:
            best_dist = dist
            best_label = label

    # 만약 아무것도 못 찾았으면 fallback
    if best_label is None:
            best_label = np.argmax(areas[1:]) + 1

    component_mask = np.zeros_like(thresh)
    component_mask[labels == best_label] = 255
    results["component_mask"] = component_mask

    # 7) FLOODFILL
    digit = component_mask.copy()
    h, w = digit.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    ys, xs = np.where(component_mask == 255)
    seed = (int(xs.mean()), int(ys.mean()))

    cv2.floodFill(digit, mask, seed, 255, (5), (5), flags=8)
    results["digit_filled"] = digit

    # 8) CROP
    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]
    crop = digit[y:y+bh, x:x+bw]
    results["crop"] = crop

    # 9) RESIZE TO 20×20 → PAD TO 28×28
    rows, cols = crop.shape
    if rows > cols:
        factor = 20 / rows
        new_rows = 20
        new_cols = int(cols * factor)
    else:
        factor = 20 / cols
        new_cols = 20
        new_rows = int(rows * factor)

    resized = cv2.resize(crop, (new_cols, new_rows))
    canvas = np.zeros((28,28), dtype=np.uint8)

    x_pad = (28 - new_cols) // 2
    y_pad = (28 - new_rows) // 2
    canvas[y_pad:y_pad+new_rows, x_pad:x_pad+new_cols] = resized
    
    results["canvas"] = canvas

    return results