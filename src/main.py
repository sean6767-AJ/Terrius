# main함수 : 주행 중 일정거리 이하면 camera start > 차량이 회전 직전까지 CNN 모델로 숫자 인식
# 일정 거리 사이에서 인식된 숫자들 중 가장 확률이 높은 숫자 출력

from camera_module import init_camera, get_frame
from preprocessing import preprocess_frame
from inference import predict_digit
import serial
import time

# 카메라 초기화
cap = init_camera()

# 아두이노 시리얼 초기화
ser = serial.Serial('COM3', 9600)
time.sleep(2) # 아두이노 시작 대기

# 메인 루프 

capturing = False # 인식 모드 off

START_THRESHOLD = "CAPTURE"

while True:

    msg = ser.readline().decode().strip()

    if msg == "CAPTURE":
        capturing = True
        predictions = []
        print("인식 시작")
    
    # 인식중
    if capturing:
        frame = get_frame(cap)
        digit = predict_digit(preprocess_frame(frame))
        predictions.append(digit)

    # 인식 종료
    if msg == "END":
        capturing = False

        if len(predictions) > 0:
            from collections import Counter # Counter는 list 안에 각 요소가 몇번 등장했는지 세줌
            final_digit = Counter(predictions).most_common(1)[0][0] # Counter 형태 [(숫자, 중복횟수)] > [0] : () > [0] > 숫자
            print("숫자는:",final_digit)

        predictions = [] # 다음 인식을 위해 리스트 비움





