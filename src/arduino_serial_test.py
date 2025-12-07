import serial
import cv2
import time
import numpy as np

# COM3 연결
ser = serial.Serial("COM3", 19200)
time.sleep(2)

print("===== 키보드 조종 모드 =====")
print("R: 오른쪽 회전")
print("L: 왼쪽 회전")
print("W: 전진")
print("B: 후진")
print("Q: 종료")
print("===========================")

# 빈 창 하나 만들어야 waitKey가 동작함
cv2.namedWindow("Controller")
blank = 255 * (np.ones((200, 200, 3), dtype=np.uint8))

while True:
    cv2.imshow("Controller", blank)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        print("오른쪽 회전")
        ser.write(b"R")

    elif key == ord('l'):
        print("왼쪽 회전")
        ser.write(b"L")

    elif key == ord('w'):
        print("전진")
        ser.write(b"W")
    
    elif key == ord('b'):
        print("후진")
        ser.write(b"B")

    elif key == ord('q'):
        print("종료")
        break
    

ser.close()
cv2.destroyAllWindows()

    



