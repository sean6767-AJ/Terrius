import cv2
import numpy as np
from scipy import stats

def preprocess_frame(frame):
    # 입력 : BGR 프레임 (카메라 원본)
    # 출력 : CNN 입력용 28x28 흑백 이미지 (numpy array)

    # BGR > GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 블러(노이즈 제거 / 부드럽게) > 이미지(배열저장)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 커널 생성(대상이 있는 픽셀을 강조) 커널을 이미지 위를 슬라이딩 하며 convolution 수행 > 숫자 윤곽 선명하게
    # 가운데 값 5 : 중심 픽셀 5배 강조, 주변 값 -1 : 주변픽셀들을 빼서 대비효과, 0 : 그 방향은 영향x
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # -1은 숫자 타입(RGB : uint8) 원본과 동일하게 저장 > 이미지(배열)
    sharp = cv2.filter2D(blur, -1, kernel) 

    # 다시 대비 강화
    Equalization = cv2.equalizeHist(sharp)

    # adaptive Threshold : 숫자 / 배경 분리 > CNN 용 흑백 이미지 만들기
    thresh = cv2.adaptiveThreshold(
        Equalization,                    # Gaussian된 gray 이미지                           
        255,                             # 최대값 : 임계값 넘는 픽셀에 적용
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 가우시안 가중치를 사용한 임계값 계산 방법
        cv2.THRESH_BINARY_INV,           # 반전 : 숫자 = 흰색, 배경 = 검정(MNIST 형태)
        11,                              # 이웃 픽셀들의 블록 크기 (블록 크기가 클수록 더 넓은 영역에서 평균 계산)
        2                                # 계산된 평균에서 뺄 값, 임계값 조정하는데 사용
    )

    # contour로 둘러쌓인 부분을 흰색으로 채워야됨 > flooFill 사용하려 했으나 contour가 완전히 닫힌 도형 아니라 실패
    # CCL(Connected Components With stats) > 덩어리로 인식
    # 이후의 Adaptive FloodFill > 도형이 닫힌 도형이 아니더라도 내부를 채워줌

    # CCL 배열 저장 형식
    # 행 : 라벨 번호(0 = 배경, 1, 2, 3... = blob(덩어리) )
    # 열 : 각 라벨의 정보 
    # cv2.CC_STAT_LEFT   # x
    # cv2.CC_STAT_TOP    # y
    # cv2.CC_STAT_WIDTH  # w
    # cv2.CC_STAT_HEIGHT # h
    # cv2.CC_STAT_AREA   # area
    num_labels, labels, s_stats, centroid = cv2.connectedComponentsWithStats(thresh, connectivity = 8)

    # 인식 안되는 경우
    if num_labels <= 1:
    # 28x28 크기의 빈(검정색) 이미지 배열을 반환
        return np.zeros((28, 28), dtype="float32")

    areas = s_stats[:, cv2.CC_STAT_AREA] # 모든 행에 대해 AREA 열만 선택


    # best_label 선택 (가장 큰거 했다가 망함)


    ###########################################################



    # 노이즈가 제거된 숫자 blob mask 생성
    # thresh와 동일한 크기의 전체가 0(검정)인 이미지 생성
    component_mask = np.zeros_like(thresh) 
    # best_label(숫자 blob부분) 위치만 255(흰색) 변경
    component_mask[labels == best_label] = 255
    
    # adaptive FloodFill : blob에 해당하는 부분의 내부를 채움(blob의 contour로 둘러쌓인 영역)    
    digit = component_mask
    
    h, w = digit.shape
    mask = np.zeros((h+2, w+2), np.uint8) # fllodFill 사용시 문법
    
    # seed는 floodFill이 어디에서부터 역역을 채워나갈지 정함   
    # 흰색(255)부분의 ys(행),xs(열) 배열 반환
    ys, xs = np.where(component_mask == 255)
    
    # 행, 열의 평균을 구해 숫자의 중심을 seed로 설정
    seed = (int(xs.mean()), int(ys.mean()))
    
    # 채워 넣을 값(흰색)
    newVal = 255
    
    # 조절(허용 색 차이) > seed 픽셀 값 기준으로 밝기가 +-20 안쪽에 있는 픽셀들만 같은 영역으로 보고 채워나감
    # 조금 어두운 부분, 조금 밝은 부분 같은 blob으로 묶음
    tolerance = 10 
    
    cv2.floodFill(
        digit,                    # 채울 이미지(원본(compnent_mask)을 바꿈)
        mask,                     # (h+2, w+2) 마스크 
        seed,                     # 시작점(x,y)
        newVal,                      # 채울값(흰색)
        (tolerance)*3,             # loDiff
        (tolerance)*3,             # upDiff
        flags = 8                 # 8방향 연결(상/하/좌/우/대각선)
    )

    # boundingRect(숫자만 포함하는 작은 사각형) crop(자르기)
    x = s_stats[best_label, cv2.CC_STAT_LEFT]    # 왼쪽 x좌표
    y = s_stats[best_label, cv2.CC_STAT_TOP]     # 위쪽 y좌표
    w = s_stats[best_label, cv2.CC_STAT_WIDTH]   # 너비(width)
    h = s_stats[best_label, cv2.CC_STAT_HEIGHT]  # 높이(height)

    # crop
    digit_cropped = digit[y:y+h, x:x+w] 

    # 20x20 비율 맞춰서 28x28 canvas에 중앙 배치

    # 긴 변을 20으로 잡고 나머지 변은 비율 유지 > 숫자 찌그러짐 방지(mnist 공식 문서)
    # rows:세로 길이, cols:가로 길이 
    rows, cols = digit_cropped.shape

    if rows > cols:
        factor = 20.0 / rows                                        # 긴 변 x 비율 = 20 > 비율(factor) = 20 / row
        rows = 20
        cols = int(round(cols * factor))                            # resize는 정수 > 보다 정확한 비율을 위해 round(반올림)활용(float 형식임으로 int로 정수형으로 설정)
        digit_resized = cv2.resize(digit_cropped, (cols, rows))     # 비율 맞춰서 resize
    else:
        factor = 20 / cols
        cols = 20
        rows = int(round(rows * factor))
        digit_resized = cv2.resize(digit_cropped, (cols, rows))

    # 28 x 28 canvas 생성
    canvas = np.zeros((28,28), dtype = np.uint8)
    
    # 여백(pad) 계산
    # 28 x 28에서 숫자 크기 rows x cols 부분을 뺀 값 pad를 반으로 나눠 왼쪽/오른쪽, 위/아래 여백을 생성
    x_pad = (28 - cols) // 2 
    y_pad = (28 - rows) // 2

    # canvas 가운데에 붙이기
    canvas[y_pad:y_pad+rows, x_pad:x_pad+cols] = digit_resized

    # CNN 압력용
    digit_norm = canvas.astype("float32") / 255.0 

    return digit_norm