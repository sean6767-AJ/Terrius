# 카메라 초기화 & 프레임 캡쳐
import cv2

# 카메라 해상도 세팅값 반환
def init_camera():
    cap = cv2.VideoCapture(0) 

    # 카메라 해상도 설정
    # 숫자 인식에 적합한 해상도인 640x480으로 설정함
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# 카메라에서 한장의 프레임을 읽어 반환
def get_frame(cap):

    # ret체크를 통해 오류 방지
    ret, frame = cap.read()
    return frame