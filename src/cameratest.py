import cv2

print("=== Camera Index Scan ===")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"카메라 {i} 열림 ✓")
        cap.release()
    else:
        print(f"카메라 {i} 없음 ✗")
