import cv2
import numpy as np
from camera_module import init_camera, get_frame
from preprocessing import preprocess_frame
from inference import predict_digit

def main():
    # 카메라 세팅
    cap = init_camera()
    print("CNN 통합 테스트 시작 (q 누르면 종료)")

    while True:
        # 1) 프레임 캡쳐
        frame = get_frame(cap)
        if frame is None:
            print("프레임 읽기 실패")
            continue

        # 2) 전처리 (28x28 흑백 이미지)
        processed = preprocess_frame(frame)

        # 3) CNN 예측
        digit = predict_digit(processed)

        # 4) 프레임에 결과 그려주기
        cv2.putText(frame, f"Pred: {digit}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 5) 원본 이미지 출력
        cv2.imshow("Camera View", frame)

        # 6) 전처리 이미지 시각화 (확대)
        show_img = (processed * 255).astype("uint8")
        show_img = cv2.resize(show_img, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Preprocessed (28x28)", show_img)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


