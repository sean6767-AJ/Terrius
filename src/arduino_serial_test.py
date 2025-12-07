import serial
import time

# 아두이노 포트 (브로는 COM3)
ser = serial.Serial("COM3", 19200, timeout=1)

print("포트 연결됨:", ser.name)

time.sleep(2)  # 아두이노 리셋 타임 기다리기

while True:
    cmd = input("R 또는 L 입력 (q 종료): ").strip().upper()

    if cmd == "Q":
        break

    if cmd not in ("R", "L"):
        print("R 또는 L만 입력해라 브로")
        continue

    ser.write(cmd.encode())
    print("보낸 명령:", cmd)


