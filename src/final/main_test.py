from Camera_set import init_camera, get_frame
from OTSU_preprocessing_test import preprocessing
from inference import predict_num
from tensorflow.keras.models import load_model

import cv2
import numpy as np

# ---------------------------------------------------------
# 여러 프레임을 받아서 안정적인 숫자 예측
# ---------------------------------------------------------
def stable_predict(model, get_frame, preprocessing, shots):

    preds = []

    for _ in range(shots):
        frame = get_frame()
        if frame is None:
            continue
        
        img = preprocessing(frame)      # (28,28) 반환
        pred = predict_num(model, img)  # 숫자 예측
        preds.append(pred)

    # 가장 많이 나온 숫자 반환
    return max(set(preds), key=preds.count)


# ---------------------------------------------------------
# 메인 테스트 코드
# ---------------------------------------------------------
def main():

    print("모델 로드 중...")
    model = load_model("D:/Terrius/models/mnist_cnn_finetuned.h5")

    print("카메라 초기화 중...")
    cap = init_camera()

    print("SPACE: 숫자 인식 시작 | Q: 종료")

    while True:

        # 카메라 프레임 얻기
        frame = get_frame(cap)
        if frame is None:
            continue

        # 실시간 원본 화면 출력
        cv2.imshow("Camera", frame)

        # 전처리 화면도 출력
        processed = preprocessing(frame)  # 0~1의 (28,28)
        digit_show = (processed * 255).astype("uint8")
        digit_big = cv2.resize(digit_show, (280, 280), cv2.INTER_NEAREST)
        cv2.imshow("Processed 28x28", digit_big)

        key = cv2.waitKey(1) & 0xFF

        # ---------------------------------------------------
        # SPACE 누르면 CNN 60프레임 예측 시작
        # ---------------------------------------------------
        if key == ord(' '):
            print("\n 숫자 인식 시작 ")

            number = stable_predict(
                model,
                lambda: get_frame(cap),
                preprocessing,
                shots=20
            )

            print(f" 최종 예측값: {number}\n")

        # ---------------------------------------------------
        # Q 누르면 종료
        # ---------------------------------------------------
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
