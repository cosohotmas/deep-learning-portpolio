# 통합 showcase — 분류와 회귀를 하나의 포트폴리오로

Part I `수강생배포_Streamlit특강/showcase`의 구조를 이 미니프로젝트에 맞게 줄였습니다.

```text
showcase/Home.py
├─ pages/1_분류_MNIST.py  ──> 과제_Streamlit_앱_분류/app.py
└─ pages/2_회귀_자전거.py ──> 과제_Streamlit_앱_회귀/app_bike.py
```

showcase는 앱을 복사하지 않고 **그대로 재사용**합니다. 두 앱은 각각 독립 실행도 가능합니다.

## 학생이 작성할 곳

1. 두 앱의 `TODO 1~4`를 완성합니다.
2. 두 노트북을 실행해 각 앱 폴더에 `.pt` 체크포인트를 생성합니다.
3. `showcase/profile.py`의 이름·분류 한계·회귀 오차·다음 실험을 내 결과로 작성합니다.

## 로컬 실행

`미니프로젝트/` 폴더에서 실행합니다.

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m streamlit run showcase/Home.py
```

브라우저에서 Home과 두 페이지를 모두 열고 다음을 확인합니다.

- Home의 프로젝트 카드와 내 설명이 보이는가
- 분류 페이지에서 업로드 이미지가 28×28 미리보기와 예측으로 이어지는가
- 회귀 페이지에서 슬라이더 변경이 예측값 변경으로 이어지는가
- 사이드바 지표가 노트북의 최종 기록과 같은가

## Streamlit Community Cloud

권장 제출은 두 앱을 묶은 **showcase URL 하나**입니다. 저장소 루트에 아래 항목을 올립니다.

```text
.streamlit/config.toml
requirements.txt
showcase/
과제_Streamlit_앱_분류/app.py
과제_Streamlit_앱_분류/mnist_cnn.pt
과제_Streamlit_앱_회귀/app_bike.py
과제_Streamlit_앱_회귀/bike_reg.pt
```

Cloud 설정:

- Main file path: `showcase/Home.py`
- Python: `3.11`
- API key / `.env`: 필요 없음

공개 URL을 직접 열어 세 페이지를 확인한 뒤 제출합니다. 배포 실패 시 두 앱을 각각 로컬로 실행한 화면과 오류 메시지를 함께 제출해 원인을 확인합니다.
