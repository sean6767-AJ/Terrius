# 영상 > 프레임단위로 끊기

import cv2

def start_camera_loop():
    # 1 > 외부 웹캠
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("Camera Loop Started (ESC 누르면 종료)")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        # 화면에 현재 프레임 띄우기
        cv2.imshow("Terrius Camera", frame)

        # ESC(27) 누르면 종료
        if cv2.waitKey(0) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()   

# 파일을 직접 실행해야 카메라 ON
if __name__ == "__main__":
    start_camera_loop()