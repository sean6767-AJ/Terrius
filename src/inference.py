# 인식된 숫자 맞추기
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model("D:/Terrius/models/mnist_cnn.h5")
print("CNN 모델 로드 완료")

def predict_digit(img):
    
    # img_28x28 : (28, 28) 또는 (28, 28, 1) 형태의 흑백이미지
    # img_28x28 전처리 완료 at preprocessing
    # 리턴값 : 예측한 숫자 (0~9)

    # 2) 차원 확장 (CNN 입력 형태 : 1, 28, 28, 1) 
    # 첫 번째 1 : batch size(몇 장을 한번에 넣을지)
    # 두 번째 1 : channels : 흑백(1), 컬러 RGB(3)
    if img.ndim == 2:
        img = np.expand_dims(img, axis = -1)
    
    img = np.expand_dims(img, axis = 0)

    # 3) 예측 실행
    pred = model.predict(img, verbose = 0)

    # 4) 가장 높은 확률의 숫자 반환
    digit = np.argmax(pred)

    return digit