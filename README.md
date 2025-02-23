# MeetGenius
[소개 영상]

![12기_공통PJT_영상_포트폴리오_B203](/uploads/0888fe43e5c860eb79aa25239167be9e/12기_공통PJT_영상_포트폴리오_B203.mp4)

## 💡 개요
- 진행기간 : 2025.01.06 ~ 2025.02.21.
- 주제 :  온디바이스 AI 회의 비서
- 서비스명 : MeetGenius
  
|   팀원    | 역할 |
|--------|-------|
|정찬호| PM, Infra, Data|
|박가연| Data, AI engineering|
|연재환| aI engineering, Prompt Engineering, Back-end 개발 (Fast API)|
|김근휘| Back-end 개발(Django, Fast API)|
|장인영| Back-end 개발(Django), Front-end 개발(React.js)|
|정유진| Front-end 개발(React.js), Design(Figma)|

<br>

##  ⚙️ 기술 스택
### Frontend
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=black)
![Redux](https://img.shields.io/badge/Redux-764ABC?style=for-the-badge&logo=Redux&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=black)
![CSS](https://img.shields.io/badge/CSS-1572B6?style=for-the-badge&logo=CSS3&logoColor=white)
![Axios](https://img.shields.io/badge/Axios-5A29E4?style=for-the-badge&logo=Axios&logoColor=white)
![Figma](https://img.shields.io/badge/Figma-F24E1E?style=for-the-badge&logo=Figma&logoColor=white)

### Backend
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=Django&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=Celery&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=FastAPI&logoColor=white)
![MariaDB](https://img.shields.io/badge/MariaDB-003545?style=for-the-badge&logo=MariaDB&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=Redis&logoColor=white)
![Postman](https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=Postman&logoColor=white)


### AI/Data
![Jetpack](https://img.shields.io/badge/Jetpack-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Firmware](https://img.shields.io/badge/Firmware-808080?style=for-the-badge&logo=&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![EXAONE](https://img.shields.io/badge/EXAONE-EC297B?style=for-the-badge&logo=&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-FFC107?style=for-the-badge&logo=&logoColor=white)

### Infra
![Docker](https://img.shields.io/badge/Docker-0db7ed?style=for-the-badge&logo=docker&logoColor=white)

### Environment
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![Mattermost](https://img.shields.io/badge/Mattermost-0058CC?style=for-the-badge&logo=mattermost&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)


<br>

## ❗ 기획 배경
### 사용자 페인 포인트 및 니즈

1. **보안성**
    - 페인포인트: 최근 기업 기밀정보의 외부 유출 사례가 증가하면서, 클라우드 서버 의존 시 보안 취약점(물리·네트워크·시스템 등)이 발생할 가능성이 높음.
    - 니즈: 기업 내부망(온프레미스)에서 데이터를 처리하여 기밀정보를 안전하게 보호하고, 보안 리스크를 최소화할 수 있는 솔루션 필요.
      
2. **비효율적인 회의 운영**
    - 페인포인트: 회의록 작성, 관련 문서 탐색, 회의 내용 정리 등의 단순 반복 업무가 많아 비효율적임.
    - 니즈: 프로젝트 및 회의 정보를 한곳에서 통합 관리하고, 회의 생산성을 높이는 자동화 솔루션 필요.

### 서비스

1. **보안 강화**
    - Jetson Orin 보드를 활용한 온디바이스 AI 회의 비서를 제공하여, 모든 데이터가 내부망에서 처리되도록 보장함.

2. **효율적인 회의 운영**
    - 반복적인 회의 기록 및 회의록 작성 업무를 자동화하여 회의 생산성을 극대화함.
    - 프로젝트와 회의 정보를 통합 관리하여 불필요한 업무를 줄이고, 실질적인 논의에 집중할 수 있도록 지원함.

3. **실시간 정보 활용**
    - 회의 도중 필요한 문서를 신속하게 검색하고, 관련 정보를 즉시 제공하여 의사결정 속도와 정확성을 향상함.
    - 회의 중 생기는 이슈나 과제에 대해 즉각적인 판단을 내릴 수 있도록 지원함.

4. **STT 기술 결합으로 사용성 제공**
    - 실시간 음성인식(STT) 기술을 활용하여 회의 내용을 자동으로 텍스트로 변환함.
    - 이를 통해 회의록 작성 시간을 줄이고, 기록된 내용을 쉽게 검색 및 분석할 수 있도록 함.

<br>

## 📄 설계 
### 아키텍처
 <img src="images/시스템 아키텍처.png">
 <img src="images/작동방법.png">

### ERD
 <img src="images/ERD.png">



<br>

## 🖥️  서비스 대표 기능
**1. 프로젝트 생성 & 문서 삽입**

- 프로젝트 생성: 프로젝트 이름, 내용, 참여자 목록을 입력하여 프로젝트 단위로 관리할 수 있음음
- 프로젝트 관련 문서 삽입: 문서는 Chroma DB에 청크 단위로 벡터화되어 저장되며, 회의 자료 탐색 RAG에 활용

![프로젝트생성_회의예약](/uploads/a367774484fe5da101963785b75de010/프로젝트생성_회의예약.mp4) 



**2. 회의 예약**
- 회의 예약: 프로젝트 참여자 목록을 자동으로 불러옴
- 실시간 회의 페이지에서 안건과 관련된 문서를 확인할 수 있음

![회의생성](/uploads/c26356d8101cf8318803f580e996c44f/회의생성.mp4)


- 삭제 권한 제한: 마스터 및 참여자 권한에 따라 삭제가 제한되며, 참여자가 아닐 경우 삭제할 수 없음

![권한관련영상](/uploads/72a776008fef777eb5833eb97fa1b3d2/권한관련영상.mp4)

   
**3. 회의 기록**
- 회의 내용을 기록하고, 실시간 회의 페이지에서 해당 내용을 제공

![회의_기록](/uploads/d9c796cd22b672c27b0b372132c06b30/회의_기록.mp4)

**4. 자료 탐색**
- AI 비서 **아리**에게 궁금한 점을 질문
- 관련 문서를 기반으로 답변 제공

![자료탐색](/uploads/f1124d9d0e051bedc44a45e2ea634892/자료탐색.mp4)

**5. 동일한 화면**
- 모든 사용자에게 동일한 회의 기록을 제공

![화면_동일](/uploads/dbc4c342be7d4534731f58ec203e538f/화면_동일.mp4)

**6. 회의록 생성**
- 회의 전체 내용을 확인하고 수정 가능
- 수정 후, AI 회의 비서가 회의록을 요약
- 회의 요약에는 약 2~3분 소요

![회의록_요약](/uploads/133b6ae37d370457a1330d0372fed613/회의록_요약.mp4)

<br>


<br>

## 소감
| 팀원     | 소감 |
|-----------|--------|
|정찬호 🐨| |
|박가연 🐥| |
|연재환 🤖| |
|김근휘 🐶| |
|장인영 🐹| |
|정유진 🐇| |