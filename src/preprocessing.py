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
        cv2.THRESH_BINARY_INV,        q   # 반전 : 숫자 = 흰색, 배경 = 검정(MNIST 형태)
        11,                              # 이웃 픽셀들의 블록 크기 (블록 크기가 클수록 더 넓은 영역에서 평균 계산)
        2                                # 계산된 평균에서 뺄 값, 임계값 조정하는데 사용
    )

    # 숫자 내부 흰색으로 채워줌
    kernel = np.ones((20,20), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


    # contour(윤곽선) 선택   
    # filter 별 score를 계산해 최종 final_score를 저장한 뒤 가장 점수가 높은 contour 선택  
    
    contour_info = [] # contour의 정보 : final_score 저장할 배열
    w, h = frame.shape[:2] # frmae 배열은 (width, height, channels) > width, height만 가져옴
    frame_area = w*h

    # contours는 여기에서 먼저 선언해야 함 (for문 밖)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours : 
        
        contour_area = cv2.contourArea(cnt)
        if contour_area == 0:
            continue
        area_norm = contour_area / frame_area # 0~1 사이(contour면적 / 전체 frame 면적)
        # 10%를 기준값으로 score = (현재크기) / (기준크기) 기준크기를 10%로 잡고 정규화
        # 10% 보다 크면 만점
        area_score = min(area_norm / 0.1, 1.0) 

        # 비율 필터링 : 숫자를 감싸는 사각형의 가로세로 비율로 숫자가 아닌 contours를 걸러냄
        # 가로(width) / 세로(heigth) = ratio > 숫자는 ratio가 0.2 ~ 0.5
        x, y, w, h = cv2.boundingRect(cnt) # x,y : 박스의 위치 w, h : 박스의 크기, contour을 포함하는 최소크기 직사각형 만들기
        ratio = w / h 
        ideal_min = 0.2
        ideal_max = 5.0 # 실제 측정에서는 ratio 범위를 넓게 설정

        if ideal_min <= ratio <= ideal_max:
            ratio_score =1.0
        elif ratio < ideal_min:
            ratio_score = ratio / ideal_min # 0~1로 정규화
        else :
            ratio_score = ideal_max / ratio
        
        # 밀집도 필터링 : solidty = contour 면적 / contour의 볼록 껍질
        # 숫자는 밀집되어 있다 > soidty 0.3 ~ 1.0 (일반적인 contour의 solidty : 0.0 ~ 0.2)
        hull = cv2.convexHull(cnt) # 볼록 껍질
        hull_area = cv2.contourArea(hull) # hull의 면적

        if hull_area == 0:
            continue

        solidty = float(contour_area) / hull_area 
        solidty_score = solidty # 0~1 값으로 잘 정규화 되어있음

        # final_score은 가중치(중요도)를 곱해 설정
        final_score = 0.5 * area_score + 0.3 * solidty_score + 0.2 * ratio_score

        # contour_info 배열에 contour 별 final_score 저장
        contour_info.append({
            "cnt" : cnt, 
            "final_score" : final_score
        })

    # contour를 못 찾으면 예외 처리 
    if len(contour_info) == 0:
        return np.zeros((28,28), dtype=np.float32)

    best = max(contour_info, key = lambda x : x["final_score"]) # 추가설명 필요..

    # best에서 cnt 부분만 꺼내 최종 contour로 저장
    best_cnt = best["cnt"]

    x, y, w, h = cv2.boundingRect(best_cnt) # 숫자 영역 좌표 계산

    digit = thresh[y:y+h, x:x+w] # 숫자를 포함하는 boundingRect 영역만 잘라냄

    # CNN 모델의 인식 향상을 위해 crop한 숫자를 20x20으로 만든 뒤 28x28 검은 배경 가운데에 붙여줌

    digit_20 = cv2.resize(digit, (20,20), interpolation=cv2.INTER_AREA) 
    canvas = np.zeros((28,28), dtype = np.uint8) # 0은 검은색, data type 설정 
    canvas[4:24, 4:24] = digit_20

    digit_norm = canvas.astype("float32") / 255.0    

    return digit_norm
