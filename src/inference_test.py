# inference_test.py
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model("D:/Terrius/models/mnist_cnn.h5")
print("CNN 모델 로드 완료")

def predict_digit(img_28x28):
    """
    img_28x28 : (28,28) 또는 (28,28,1) 형태의 흑백이미지
    return    : 예측된 숫자 (0~9), 실패하면 -1
    """

    # 전처리 실패 or 잘못된 입력이면 -1 반환
    if img_28x28 is None:
        return -1

    # numpy array 보장
    img = np.array(img_28x28)

    # 혹시 전체가 검정 → 숫자 없음 → -1
    if img.sum() == 0:
        return -1

    # shape이 (28,28)이면 reshape
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    # CNN 입력 형태(1,28,28,1)
    img = np.expand_dims(img, axis=0)

    # 예측
    pred = model.predict(img, verbose=0)
    digit = int(np.argmax(pred))

    return digit
