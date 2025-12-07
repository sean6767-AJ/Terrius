import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 너의 전처리 함수 import
from final.preprocessing_OTSU_real_real_final import preprocessing


# ========================
# 1) 데이터 로딩 함수
# ========================
def load_real_dataset(base_dir):
    X = []
    Y = []

    for label in range(10):
        folder = os.path.join(base_dir, str(label))
        if not os.path.exists(folder):
            continue
        
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):

                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                # 네가 쓰는 실제 MNIST 방식 preprocessing
                img28 = preprocessing(img)  # (28,28) / 0~1 float32

                X.append(img28)
                Y.append(label)

    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, 28, 28, 1)
    Y = to_categorical(Y, 10)

    return X, Y


# ========================
# 2) 모델 로드 + 미세조정 준비
# ========================
def finetune_model():
    real_data_dir = r"D:/Terrius/real_data"
    model_path = r"D:/Terrius/models/mnist_cnn_finetuned.h5"   # 기존 모델 불러오기
    save_path  = r"D:/Terrius/models/mnist_cnn_finetuned3.h5"  # 새로운 모델로 저장

    print("📌 Loading real dataset...")
    X, Y = load_real_dataset(real_data_dir)
    print(f"총 {len(X)}장 로드됨.")

    # 모델 로드
    print("📌 Loading base model...")
    model = load_model(model_path)

    # 미세 조정용 러닝레이트 ↓↓↓ (아주 작게!)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.00005),
        metrics=["accuracy"]
    )

    # ========================
    # 3) 학습
    # ========================
    print("📌 Start Fine-tuning...")
    history = model.fit(
        X, Y,
        batch_size=8,
        epochs=15,
        shuffle=True,
        verbose=1
    )

    # ========================
    # 4) 저장
    # ========================
    print("📌 Saving fine-tuned model...")
    model.save(save_path)
    print("완료! →", save_path)


if __name__ == "__main__":
    finetune_model()





