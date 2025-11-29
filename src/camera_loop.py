# 카메라 초기화 & 프레임 캡쳐

# 카메라 해상도 세팅값 반환
def init_camera() :
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap
# 카메라에서 한장의 프레임을 읽어 반환
def get_frame(cap)
    ret, frame = cap.read() # ret은 프레임을 제대로 읽었는지 확인
    return frame