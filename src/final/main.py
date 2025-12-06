from Camera_set import init_camera
from Camera_set import get_frame
from preprocessing_OTSU_test import preprocessing
from inference import predict_num
from tensorflow.keras.models import load_model
import cv2
import serial

# 아두이노 통신
ser = serial.Serial('COM3', 9600)

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
    
    # start_flag = True > CNN 숫자인식 시작
    start_flag = False 

    print("start")

    # 아두이노 회전 제어
    # 회전 순서가 고정되어 있기 때문에 step값만 조절해주면 turn_R, turn_L 제어 가능 
    # 숫자 인식 > R & L값 아두이노 송신 > 회전
    commands = ["R", "L"]
    step = 0

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
        
        # start_flag 트리거 스페이스바
        if key == ord(' '):
             start_flag = True
        
        # 숫자 인식 시작
        if start_flag:
            print("숫자 인식 시작")

            # lambda 함수는 get_fame() shot마다 호출
            number = stable_predict(model, lambda : get_frame(cap), preprocessing, shots = 20)
            
            print("예측된 숫자는:", number)

            # 첫번 째 인식 끝 step = 0
            # commands[step] = R > 아두이노 오른쪽 회전 제어
            # 두번 째 인식 끝 step = 1
            # commands[step] = L > 아두이노 왼쪽 회전 제어
            cmd = commands[step]
            ser.write(f"{cmd}\n".encode())

            step += 1

            # 다음 숫자 인식을 위해 start_flag off
            start_flag = False
        
        # q 누르면 종료
        if key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

    if __name__ == "__main__": 
        main()







