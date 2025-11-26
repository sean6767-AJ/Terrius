# 전처리 과정

import cv2
import numpy as np

def preprocess_frame(frame):
    # 입력 : BGR 프레임 (카메라 원본)
    # 출력 : CNN 입력용 28x28 흑백 이미지 (numpy array)

    # 1) BGR > GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) 블러(노이즈 제거)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Thresholding (이진화)
    # 숫자 배경에 따라 조절해야 할 수도 있음
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 4) Contour 찾기(숫자 모양이 가장 큰 영역)
    contours, _ = cv2.findContours (
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        # 인식 실패 > 빈 28x28 반환해서 CNN의 오작동 방지
        return np.zeros((28,28), dtype = np.float32)
    
    # 5) 가장 큰 컨투어(숫자 영역) 선택
    cnt = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # 6) crop (숫자 부분만 자르기)
    digit = thresh[y:y+h, x:x+w]

    # 7) 28x28 resize (MNIST 입력 크기)
    digit_resized = cv2.resize(digit, (28,28))

    # 8) 0~1 정규화
    digit_norm = digit_resized.astype('float') / 255.0

    return digit_norm