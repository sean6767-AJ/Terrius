# test_mnist_inference.py

import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# -------------------------------------------------
# 1) 학습된 모델 로드
# -------------------------------------------------
model = load_model("D:/Terrius/models/mnist_cnn.h5")


def mnist_random_test(model, x_test, y_test, num_samples=1000):
    correct = 0

    for _ in range(num_samples):
        idx = random.randint(0, len(x_test)-1)

        img = x_test[idx]      # (28, 28, 1)
        label = y_test[idx]    # 0~9 정답

        img_input = np.expand_dims(img, axis=0)  # (1,28,28,1)

        pred = model.predict(img_input, verbose=0)
        digit = np.argmax(pred)

        if digit == label:
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 테스트 개수: {num_samples}")
    print(f"🔥 랜덤 샘플 정확도: {accuracy * 100:.2f}%")
    return accuracy

# -------------------------------------------------
# 2) MNIST 테스트셋 불러오기
# -------------------------------------------------
(_, _), (x_test, y_test) = mnist.load_data()

# 전처리: CNN 입력 형태 맞추기
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# -------------------------------------------------
# 3) 랜덤 테스트 실행
# -------------------------------------------------
mnist_random_test(model, x_test, y_test, num_samples=1000)

