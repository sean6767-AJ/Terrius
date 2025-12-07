import cv2
import os   
import numpy as np

from Camera_set import init_camera, get_frame
from preprocessing_OTSU_real_final import preprocessing

SAVE_ROOT = r"D:\Terrius\real_data"  # 원본 이미지 저장 폴더

def main():

    print("카메라 초기화 중...")
    cap = init_camera()
    os.makedirs(SAVE_ROOT, exist_ok=True)

    save_count = 0

    print("=== 데이터 수집 모드 ===")
    print("SPACE → 원본 이미지 저장")
    print("Q → 종료")

    while True:

        # 카메라 프레임 얻기
        frame = get_frame(cap)
        if frame is None:
            continue

        # 원본 출력
        cv2.imshow("Camera (Original)", frame)

        # 전처리 화면도 보기만 (저장 안 함)
        processed = preprocessing(frame)
        digit_show = (processed * 255).astype("uint8")
        digit_big = cv2.resize(digit_show, (280, 280), cv2.INTER_NEAREST)
        cv2.imshow("Processed 28x28", digit_big)

        key = cv2.waitKey(1) & 0xFF

        # ============================
        #   🔥 SPACE → 원본 이미지 저장
        # ============================
        if key == ord(' '):
            filename = os.path.join(SAVE_ROOT, f"raw_{save_count:05d}.png")
            cv2.imwrite(filename, frame)
            print(f"[원본 저장 완료] {filename}")
            save_count += 1

        # ============================
        #   🔥 Q → 종료
        # ============================
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
