from Camera_set import init_camera
from Camera_set import get_frame
from preprocessing_OTSU_test import preprocessing
from arduino_serial_test import predict_num
from tensorflow.keras.models import load_model
import cv2
import serial
import time

# 아두이노 통신
ser = serial.Serial('COM3', 19200) # 직접 test 해보니 컴퓨터 시리얼 통신은 19200
time.sleep(2) 

# 숫자 인식을 한장의 frame으로 하는 것은 정확하지 않을 수 있음
# CNN 모델에 여러장의 frame을 입력해 그 반환값 중 가장 빈도수가 높은 값을 반환
def stable_predict(model, get_frame, preprocessing, shots = 60):
    """
    model : CNN 모델
    get_frame : frame 읽어오는 함수(get_frame(cap))
    preprocessing : 전처리 함수
    shots : 예측 frame
    """

    # 예측값 저장할 list
    preds = []

    for _ in range(shots):
        frame = get_frame()
        img = preprocessing(frame)       # 전처리
        pred = predict_num(model, img)   # 숫자 예측
        preds.append(pred)

    return max(set(preds), key=preds.count) # 빈도수 가장 높은 값 반환

def main():

    print("모델 로드 중")
    model = load_model("D:/Terrius/models/mnist_cnn_finetuned.h5")

    print("카메라 초기화 중")
    cap = init_camera()
    
    print("start")
    print("r > 오른쪽 회전")
    print("l > 왼쪽 회전")
    print("W > 전진")
    print("s > 숫자 인식 수행")
    print("q > 종료")

    while True:
        frame = get_frame(cap)
        if frame is None:
            continue
        
        # 실시간 카메라 화면 & 전처리 화면 띄우기
        processed = preprocessing(frame)

        cv2.imshow("Camera", frame)

        # Camera랑 사이즈가 동일하게 resize
        digit_show = (processed * 255).astype("uint8")
        digit_big = cv2.resize(digit_show, (280, 280), cv2.INTER_NEAREST)
        cv2.imshow("Processed 28x28", digit_big)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
             print("오른쪽 회전")
             ser.write(b"R")

        if key == ord('l'):
             print("왼쪽 회전")
             ser.write(b"L")

        if key == ord('w'):
             print("전진")
             ser.write(b"W")
    
        # 숫자 인식 
        if key == ord('s'):
            print("숫자 인식 시작")

            # lambda 함수는 get_fame() shot마다 호출
            number = stable_predict(model, lambda : get_frame(cap), preprocessing, shots = 20)
            
            print("예측된 숫자는:", number)

        # q 누르면 종료
        if key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

    if __name__ == "__main__": 
        main()







