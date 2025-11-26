# camera_loop_test.py
import cv2
import numpy as np
from preprocessing import preprocess_frame
from inference_test import predict_digit   # 예측 결과도 보고 싶다면

def start_camera_loop_test():
    cap = cv2.VideoCapture(0)   

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("🎥 Camera Loop Test Started (ESC 눌러 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 전처리 수행
        processed = preprocess_frame(frame)   # (28,28) 형태

        # CNN 예측 (optional)
        digit = predict_digit(processed)

        # 전처리 이미지를 보기 좋게 키워서 표시
        show_processed = cv2.resize(processed, (280, 280), interpolation=cv2.INTER_NEAREST)

        # 원본 프레임에 예측 결과 표시
        cv2.putText(frame, f"Digit: {digit}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2)

        # 화면 출력(두 창)
        cv2.imshow("Terrius Camera", frame)
        cv2.imshow("Preprocessed (28x28)", show_processed)

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera_loop_test()
