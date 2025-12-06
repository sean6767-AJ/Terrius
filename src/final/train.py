from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from preprocessing_OTSU import preprocessing

# 1. 모델 로드
model = load_model("D:/Terrius/models/mnist_cnn.h5")

# 2. 실사 데이터 로드 함수
def load_real_dataset(root_dir):
    images = []
    labels = []

    for digit in range(10):
        folder = os.path.join(root_dir, str(digit))
        
        if not os.path.isdir(folder):
            print("폴더 없음:", folder)
            continue
        
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)

            img = cv2.imread(path)
            if img is None:
                print("이미지 로드 실패:", path)
                continue

            processed = preprocessing(img)     # (28,28)
            processed = processed.reshape(28, 28, 1)   # CNN 입력 형태

            images.append(processed)
            labels.append(digit)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


# 3. 실사 데이터 불러오기
X_real, y_real = load_real_dataset("D:/Terrius/real_data")
print("실사 데이터 shape:", X_real.shape, y_real.shape)


# 4. 모델 fine-tuning
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_real, y_real,
    batch_size=4,
    epochs=5,
    shuffle=True
)

# 5. 저장
model.save("mnist_cnn_finetuned.h5")

print("🔥 Fine-tuning 완료!")




