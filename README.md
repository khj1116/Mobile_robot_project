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
---
## System Architecture
### ![Image](https://github.com/user-attachments/assets/a802b279-f6e9-47c4-b387-2384f48a1b0e)
+ 매장이나 원격 주문 페이지에서 주문 시 주문 데이터가 로봇 팔 제어 코드로 송신됨
+ 포장 주문의 경우 Easy OCR로 Pick Up Zone의 Text를 인식하여 적재할 위치 좌표값 장소에 제조한 아이스크림을 플레이팅
+ 제조 과정 중 손이 난입할 시 yolov8로 Object Detecting 작업한 안전진단 기능이 구동되며 로봇 팔 정지
+ 로봇팔 동작 순서를 수정하여 제조 순서를 주문 시 선택 가능하도록 함(토핑 먼저 혹은 아이스크림 먼저) (xArm-Python-SDK 코드 수정)
+ 회원 주문 시 최근 주문한 3개의 메뉴를 추천메뉴로 창을 띄우는 추천 메뉴 기능을 추가(DB 주문 내역 테이블에서 조회)
---
## Results
+ FRONT_END
  + 다양한 인증 옵션 제공(ID/PW,FACE-ID)
  + 접근성 강화 : 다국어, 큰 글씨 모드, 다중 페이지 구현(매장/포장/회원/비회원)
  + 시각적 효과 : 물결 애니메이션으로 사용자 경험 개선
  + 병원 회진 로봇 "HOSPY" 의료진 회진 시 사용된 2D LIDAR TRACKING
+ BACK_END
  + 실시간 주문 관리 : ROS2 및 Socket I.O로 로봇팔과 클라이언트 동기화
  + 사용자 인증 : 다양한 로그인 옵션과 보안 강화(비밀번호 해싱, 세션)
  + 주문 동기화 : 포장 주문 서버와 메인 서버 간 데이터 통합
 

+ FACE ID 주문


![Image](https://github.com/user-attachments/assets/6425bd51-bd0d-420b-b912-07c3d897a0d4)


+ 큰 글씨모드 제공을 통한 사용자 경험 제공


![Image](https://github.com/user-attachments/assets/ba3b9ccc-ad7e-40a2-819c-87a3030cdf9b)
+ 매장 주문 페이지 주문 및 실시간 주문 내역 페이지 연동(매장 빌지 역할)


![Image](https://github.com/user-attachments/assets/36bb9609-c7be-4d89-b90c-352103ce6fec)


+ 영어 지원을 통한 사용자 경험 제공


![Image](https://github.com/user-attachments/assets/21be9ee2-b53f-4bad-aef0-fc5f4f45b9ce)
+ 포장 주문 페이지 주문 및 실시간 주문 내역 페이지 연동


https://github.com/user-attachments/assets/e7b77822-29d7-4137-9442-1265355ab86d

+ 포장 주문 후 아이스크림 제조

https://github.com/user-attachments/assets/3fba2deb-5ba2-4692-ad41-b80691cccf38
---
## Improvements
+ FRONT_END
  + 다국어 지원 : 추가 언어 확장 검토
  + 보안 : HTTPS 적용 및 입력값 검증 강화 필요
+ BACK_END
  + 얼굴 인식 : Python 스크립트 실행 속도 및 안정성 개선 필요
  + 보안 : HTTPS 적용, SQL 인젝션 방지 강화
  + 성능 : 대용량 주문 데이터 처리 시 DB 성능 최적화 필요
  + ROS2 : 에러 처리 및 재연결 로직 추가 검토
+ FACE-ID
  + 웹캠 품질 및 조명 조건에 따른 인식률 개선
  + DeepFace 모델 성능 최적화(GPU 메모리 사용량, 임계값 조정)


