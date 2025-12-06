# 인식된 숫자 맞추기
import numpy as np

# preprocessing에서 전처리된 img를 CNN모델로 예측하여 가장 높은 확률의 숫자 반환
def predict_num(model, img):
    
    # 차원 확장 (CNN 입력 형태 : batch, height, width, channel) 
    # batch size = 1 : 한 장만 예측
    # channel = 1 : 흑백 이미지
    
    # channel 차원 추가
    if img.ndim == 2:
        img = np.expand_dims(img, axis = -1)
    
    # batch 차원 추가
    img = np.expand_dims(img, axis = 0)

    # 예측(pred는 10개의 확률 벡터 형태)
    # 실시간 영상 처리의 효율성을 위해 vebose(로그) = 0으로 설정
    pred = model.predict(img, verbose = 0)

    # 가장 높은 확률의 숫자 반환
    num = np.argmax(pred)

    return num