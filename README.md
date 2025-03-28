# 애드인에듀 로봇 AI 개발자 부트캠프 심화과정 3기
## Overview
### 현재 의료 인력 부족으로 인한 업무 과중이 지속되고 있음. 의료진들이 본연의 업무에 더욱 
### 집중할 수 있도록, 대체 가능한 업무를 수행하는 자율주행 로봇을 개발하고자 주제를 선정함
---
## Development Goals
1. 회진에 필요한 물품을 적재하고, 환자의 상태를 수시로 체크하여 DB에 기록
2. 로봇이 직접 환자와 상호작용을 함으로서, 의료진의 업무 과중을 경감시키는 효과 기대
---
## Skills
+ Integration
  + ROS2
  + Node.js
    
+ Navigation
  + Nav2
  + ROS2
    
+ DB & Web
  + MySQL
  + HTML5
  + CSS3
  + Node.js(Express)
    
+ Object Detection
  + Pytorch
  + Opencv
  + YOLOv8
    
+ Human Tracking(My role)
  + Python
  + Opencv
  + Scikit-learn library
    
+ Voice chat
  + chat GPT, Clova API
  + picovoice porcupine
---
## System Architecture
### ![Image](https://github.com/user-attachments/assets/262f5f14-6bca-47e4-8d1d-fe531cd4f9ae)
+ 기능은 크게 의료진 회진, 수시 회진, 호출 3가지로 분류
+ 의료진 회진 : 2D lidar 만을 사용하여 사람의 다리 쌍 감지하여 의료진을 트래킹하며 회진을 보조하는 기능 수행
+ Voice chat 기능을 활용하여 트래킹을 시작 및 종료 할 수 있음
+ 수시 회진 :  병실의 환자를 Detection 후 자율 주행 수행. IR 센서로 환자의 체온을 파악하며 환자의 상태 파악
+ 병실 침대의 QR를 스캔하여 환자 스케줄 관리 및 Voice chat으로 환자에게 알림 서비스 제공
+ 호출 : 병실의 QR 스캔을 통해 해당 위치로 로봇 호출이 가능하여 여러가지 서비스를 요청할 수 있음
+ DB & Web의 경우 환자 관련 데이터(체온, 스케줄, 개인정보, 질병 관련 데이터)와 비품 데이터가 저장되며
+ 웹 인터페이스로 사용자가 간편하게 DB 내용이 수정 가능하게 함
+ WebSocket 활용하여 실시간 데이터 업데이트
+ 실제 의료기록데이터보관기간(10년)을 넘기면 자동으로 데이터가 삭제되는 기능 구현현
---
## Results
+ 로봇 호출
  + QR Scan과 ROS 통신을 원활히 연결하여 정확한 waypoint로의 주행 완성
  + NLP, STT, TTS를 활용하여 간단한 질의 응답, 음성으로 환자와 상호 작용이 가능하게 끔 구현함
  + 시각적 효과 : 물결 애니메이션으로 사용자 경험 개선
  + 병원 회진 로봇 "HOSPY" 의료진 회진 시 사용된 2D LIDAR TRACKING
+ 수시 회진 
  + TTS를 활용하여 환자와의 상호작용 기능 완성
  + Object Detection으로 환자 유무 파악 및 IR 센서로 환자 체온 체크 기능 구현
  + QR SCAN을 통한 스케줄 관리 및 DB DATA 대조 기능 구현
+ 의료진 회진
  + WakeWord, STT, TTS를 이용한 Tracking 활성화 기능 구현
  + 사람과 일정 간격을 두고 Tracking 수행
  + 추종 객체와 거리가 멀어지면 속도를 빠르게 가까우면 속도를 느리게 구현
  + 다른 사람 난입 시 정지하는 안전진단 기능 완성


+ DBSCAN 다리 쌍 감지
  + 라이다 데이터를 기반으로 접군(Points)을 클러스터링하여 다리 쌍 식별
  + 노이즈 제거 및 다리 크기(0.05m ~ 0.15m)와 점 개수(5 ~ 15개)로 장애물 필터링

![Image](https://github.com/user-attachments/assets/f70bc9f9-ae2a-4732-8e86-57be7f170a79)



+ 유클리드 거리 이용한 일정 간격 주행
  + 로봇과 목표 사이의 유클리드 거리 계산.
  + 거리와 안전 거리 차이를 기반으로 속도를 비례적으로 조정함
  + 유클리드 거리는 로봇과 사람 간 직선 거리를 나타내며, 선속도(linear.x)를 비례적으로 조정함
  + 거리가 멀수록 속도가 증가하고(최대 0.6m/s), 가까워질수록 속도가 감소하며(최소 0.05m/s),
  + 안전 거리(0.44m) 이내에서는 정지함


![Image](https://github.com/user-attachments/assets/3e850348-ae65-487c-89fd-9c33247cf714)



+ 사람 다리 감지(반원 형태) + 다리 쌍 중심 Marker 표시시


![Image](https://github.com/user-attachments/assets/9de07419-af2f-48b7-96fa-dcd39ee8ab38)


![Image](https://github.com/user-attachments/assets/f79c1f44-67ed-40de-aa8d-d30da5fb43b8)


+ 트래킹


![Image](https://github.com/user-attachments/assets/ca437451-b9c7-4d11-85b8-9e120e44992b)


+ 난입


![Image](https://github.com/user-attachments/assets/d1141c94-b8ac-4acd-9095-978c1de30456)



+ 안전진단
  + LiDAR 데이터에서 난입자 감지 및 기존 추종 대상 복원
  + 각도 범위(45도 ~ 135도) 내 점 분석, 35cm 이내 점 감지 시 난입자로 간주
  + 난입자 제거 후 기존 추종 대상 복원 로직 포함

![Image](https://github.com/user-attachments/assets/6b04b1ee-845f-486b-ae55-a2d85c53466d)

---
## Improvements
+ Human Tracking
  + Moving Obstacle Avoidance Path-Planning :  난입자나 움직이는 객체를 인식하고 스스로 회피 경로를 생성하여 기존 추종 객체를 찾아내는 알고리즘 개선
  + 의료진 Detection : Tracking 시작 시 다른 사람이 아닌 의료진을 구분하여 감지 후 트래킹(ex. 의료진 카드 인식 or QR 인식 기능)
  


