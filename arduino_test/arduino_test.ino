// DC모터 핀 설정
// MotorA - 오른쪽
#define ENA 3
#define IN1 4
#define IN2 5

// MotorB - 왼쪽
#define ENB 11
#define IN3 12
#define IN4 13

void setup() {

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(19200); // 아두이노 - pc 통신용
}

void Turn_Left(){
  int speed = 250;
  int duration = 530;
  
  analogWrite(ENA, speed); // test 필요
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  analogWrite(ENB, speed);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  delay(duration);

  stop_motor();
}

void Turn_Right(){
  int duration = 560;
  
  analogWrite(ENA, 110); 
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  analogWrite(ENB, 250);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  delay(duration);

  stop_motor();
}

void stop_motor() {
  analogWrite(ENA, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  analogWrite(ENB, 0);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'R') Turn_Right();
    if (c == 'L') Turn_Left();
  }
}



