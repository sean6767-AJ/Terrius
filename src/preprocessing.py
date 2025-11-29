# 전처리 과정

import cv2
import numpy as np

def preprocess_frame(frame):
    # 입력 : BGR 프레임 (카메라 원본)
    # 출력 : CNN 입력용 28x28 흑백 이미지 (numpy array)

    # BGR > GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 블러(노이즈 제거)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # adaptive Threshold : 숫자 / 배경 분리 > CNN 용 흑백 이미지 만들기
    thresh = cv2.adaptiveThreshold(
        blur,                            # Gaussian된 gray 이미지                           
        255,                             # 최대값 : 임계값 넘는 픽셀에 적용
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 가우시안 가중치를 사용한 임계값 계산 방법
        cv2.THRESH_BINARY_INV,           # 반전 : 숫자 = 흰색, 배경 = 검정(MNIST 형태)
        11,                              # 이웃 픽셀들의 블록 크기 (블록 크기가 클수록 더 넓은 영역에서 평균 계산)
        2                                # 계산된 평균에서 뺄 값, 임계값 조정하는데 사용
    )

    # contour(윤곽선) 선택(숫자일 확률이 높은 contour만 고르기)        
    
    # 면적 필터링 : 너무 작거나(먼지, 노이즈), 너무 큰(배경) contour 제거
    # 최소 : 전체 면적 0.2%
    # 최대 : 전체 면적 30%
    w, h = frame.shape[:2] # frmae 배열은 (width, height, channels) > width, height만 가져옴
    frame_area = w*h
    min_area = frame_area * 0.002 
    max_area = frame_area * 0.3

    main_contours = [] # contours를 저장해 전처리 과정 진행 후 CNN 모델에 반환 

    for cnt in contours : 
        
        area = cv2.contourArea(cnt)

        # 필터링 적용
        if area < min_area or area > max_area:
            
            continue # 후보 contour 아니면 버림

        # 비율 필터링 : 숫자를 감싸는 사각형의 가로세로 비율로 숫자가 아닌 contours를 걸러냄
        # 가로(width) / 세로(heigth) = ratio > 숫자는 ratio가 0.2 ~ 0.5
        x, y, w, h = cv2.boundingRect(cnt) # x,y : 박스의 위치 w, h : 박스의 크기
        ratio = w / h

        if ratio < 0.2 or ratio > 5.0 :
            continue # 후보 contour 아니면 버림

        # 밀집도 필터링 : solidty = contour 면적 / contour의 볼록 껍질
        # 숫자는 밀집되어 있다 > soidty 0.3 ~ 1.0 (일반적인 contour의 solidty : 0.0 ~ 0.2)
        hull = cv2.convexHull(cnt) # 볼록 껍질
        hull_area = cv2.contourArea(hull) # hull의 면적

        solidty = float(area) / hull_area # float는 정수 나눗셈으로 인한 오류 방지

        if solidty < 0.3:
            continue # 후보 contour 아니면 버림

        main_contours.append(cnt) # 후보 contours main에 추가













    return digit_norm