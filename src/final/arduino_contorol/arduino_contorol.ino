enum State{
  MOVING,
  STOPPED,
  COMMAND,
};

State state = MOVING;
char received_cmd = 0;


// 초음파 센서 echo, trig 핀 설정
#define echo 6
#define trig 7

// DC모터 핀 설정
// MotorA - 오른쪽
#define ENA 3
#define IN1 4
#define IN2 5

// MotorB - 왼쪽
#define ENB 11
#define IN3 12
#define IN4 13

// 초음파 센서, DC모터 핀 모드 설정
void setup() {
  pinMode(echo, INPUT);
  pinMode(trig, OUTPUT);
  
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(9600); // 아두이노 - pc 통신용
}

// 초음파 센서 모듈 값을 읽어오고 거리값을 반환 하는 함수.
float read_Ultrasonic(void) {
  
  float return_time, distance; 
  // return_time: 초음파 센서 모듈(echo) 핀이 HIGH로 유지되는 시간을 저장하는 변수(왕복시간)
  // distance: return_time값과 소리의 속도값을 적절한 계산을 통해 물체와의 거리를 계산해 저장하는 변수
  
  digitalWrite(trig, LOW); // loop() 돌릴 때 오류 방지
  delayMicroseconds(2);
  digitalWrite(trig, HIGH); // 초음파 발사
  delayMicroseconds(10); 
  digitalWrite(trig,LOW); // 초음파 발사 정지
  
  // echo 핀 HIGH 시간 측정(µs 단위)
  return_time = pulseIn(echo, HIGH);

  // 거리 계산: (음속 0.034cm/us) × 시간 / 2
  distance = (return_time*0.034)/2; 
  
  return distance; // 거리(cm)값 반환
}

void move_forward() {
  
  analogWrite(ENA, 100); // test 필요
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  analogWrite(ENB, 100);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void stop_motor() {
  
  analogWrite(ENA, 0); 
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  analogWrite(ENB, 0);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
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
}

void Turn_Right(){
  int speed = 250;     // 바퀴 속도
  int duration = 530;
  
  analogWrite(ENA, speed); 
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  analogWrite(ENB, speed);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  delay(duration);
}

// serial 통신 함수

void check_SerialCommand() {
  
  if (Serial.available()) {

    char c = Serial.read();

    if (state == STOPPED && (c == 'R' || c == 'L')) {
      received_cmd = c;
      state = COMMAND; // COMMAND 상태 전환
    }

    if (c == 'W') {
      state = MOVING;
      return;
    }
  }
}

void moving_state() {
  
  int distance = read_Ultrasonic();
  
  if (distance < 20){
    stop_motor();
    state = STOPPED;
    return;
  }

  move_forward();

}

void stopped_state() {
}

void command_state(char cmd) {
  
  if (cmd == 'R') {
    Turn_Right();
  }
  else if (cmd == 'L') {
    Turn_Left();
  }

  delay(300);
  move_forward();
  
  state = MOVING; // 회전 후 전진 상태

}

void loop() {

  // serial 처리
  check_SerialCommand();

  // state에 맞는 행동 실행
  switch(state) {

    case MOVING:
    moving_state();
    break;

    case STOPPED:
    stopped_state();
    break;

    case COMMAND:
    command_state(received_cmd);
    break;
  
  }
}



























