import cv2
import numpy as np
from preprocessing_OTSU_debug import preprocessing_debug  # 네 전처리 함수 불러오기
import importlib 

def visualize_preprocessing():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error")
            break

        # 중간 결과 전부 얻기
        r = preprocessing_debug(frame)

        # 1) GRAY
        cv2.imshow("1) GRAY", r["gray"])
        cv2.moveWindow("1) GRAY", 0, 0)

        # 2) BLUR
        cv2.imshow("2) BLUR", r["blur"])
        cv2.moveWindow("2) BLUR", 500, 0)
        
        # 3) SHARP
        cv2.imshow("3) SHARP", r["sharp"])
        cv2.moveWindow("3) SHARP", 1000, 0)

        # 4) THRESHOLD
        cv2.imshow("5) THRESH", r["thresh"])
        cv2.moveWindow("5) THRESH", 1500, 0)
        
        # 5) CROP (숫자 잘린 부분)
        cv2.imshow("8) CROP", r["digit_crop"])
        cv2.moveWindow("8) CROP", 500, 500)
        
        # 6) FINAL 28x28 CANVAS
        digit_big = cv2.resize(r["canvas"], (300,300), cv2.INTER_NEAREST)
        cv2.imshow("9) FINAL 28×28", digit_big)
        cv2.moveWindow("9) FINAL 28×28", 1000, 500)
        
        # 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_preprocessing()
