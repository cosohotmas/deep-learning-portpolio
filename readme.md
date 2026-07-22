# 수강생 배포 안내 — 미니프로젝트 1·2 + 통합 showcase

## 강사가 수강생에게 배포할 것

`미니프로젝트/` 폴더 전체를 배포합니다. `미니프로젝트_정답/`은 제출 마감 전까지 강사 전용입니다.

```text
미니프로젝트/
├─ 미니프로젝트_1_분류_MNIST_CNN_배포.ipynb
├─ 미니프로젝트_2_회귀_자전거_수요.ipynb
├─ 과제_Streamlit_앱_분류/
│  ├─ app.py                 ← 학생이 TODO 1~4 완성
│  ├─ requirements.txt
│  ├─ 배포_가이드.md
│  └─ .gitignore
├─ 과제_Streamlit_앱_회귀/
│  ├─ app_bike.py            ← 학생이 TODO 1~4 완성
│  ├─ requirements.txt
│  ├─ 배포_가이드.md
│  └─ .gitignore
├─ showcase/
│  ├─ Home.py
│  ├─ profile.py             ← 학생이 프로젝트 설명 작성
│  ├─ pages/                 ← 두 앱을 재사용하는 페이지
│  ├─ core/                  ← 테마·연결 보조 코드
│  └─ README.md
├─ requirements.txt          ← 통합 showcase 실행 환경
├─ .streamlit/config.toml
├─ 수강생_배포안내.md
└─ .gitignore
```

`mnist_cnn.pt`와 `bike_reg.pt`는 수강생이 노트북을 실행해 직접 생성합니다. `seoul_bike.csv`도 첫 실행 때 내려받으므로 초기 배포본에 없어도 정상입니다.

## 어디까지 제공하고, 어디부터 학생이 작성하는가

| 영역 | 제공 | 학생이 직접 작성·설명 |
|---|---|---|
| 노트북 | 데이터 로드·평가 뼈대·실험 함수 | 전처리·훈련 루프·모델/손실·validation 실험·최종 해석 |
| 분류 앱 | 이미지 입력·사진 정렬·결과 UI | CNN 구조·체크포인트 복원·tensor 변환·확률 추론 |
| 회귀 앱 | 슬라이더·결과 UI·체크포인트 계약 | 모델 구조·가중치 복원·특성 순서/표준화·값 추론 |
| showcase | Home·카드·두 페이지 연결 | 이름·두 모델의 한계/오차·다음 실험 |

화면을 처음부터 디자인하는 과제가 아닙니다. **학습한 모델을 앱의 입력→전처리→추론→설명으로 연결하는 것**이 평가 대상입니다.

## 권장 작업 순서

1. 분류 노트북 TODO와 validation 실험을 완료하고 `mnist_cnn.pt`를 생성합니다.
2. `과제_Streamlit_앱_분류/app.py` TODO 1~4를 완성해 독립 실행합니다.
3. 회귀 노트북 TODO와 validation 실험을 완료하고 `bike_reg.pt`를 생성합니다.
4. `과제_Streamlit_앱_회귀/app_bike.py` TODO 1~4를 완성해 독립 실행합니다.
5. `showcase/profile.py`를 내 결과로 수정한 뒤 통합 앱을 실행합니다.

```bash
cd 미니프로젝트
python3.11 -m pip install -r requirements.txt
python3.11 -m streamlit run showcase/Home.py
```

## 수강생 최종 제출물

**제출물은 아래 링크 1개입니다.** 노트북·리더보드·회고는 제출하지 않습니다 — 수업 중 활동과 자기 점검용으로만 활용합니다.

| 제출물 | 확인 내용 |
|---|---|
| Streamlit 앱 링크 | Streamlit Cloud URL, 또는 본인 포트폴리오 사이트에 연결해 둔 Streamlit 앱 링크 — Home + 분류 + 회귀 세 페이지가 모두 열림 |

권장 배포 파일과 Main file path는 `showcase/README.md`를 따릅니다. `.env`·API key·다른 수업 파일은 올리지 않습니다.

## 강사 배포 전 확인

- 학생 앱에는 TODO 1~4가 남아 있고 완성 추론 코드가 없는가
- 정답 폴더가 학생 배포 압축파일에 포함되지 않았는가
- 두 노트북이 새 폴더명으로 체크포인트를 저장하는가
- 정답 앱 두 개와 통합 showcase를 Python 3.11에서 별도 확인했는가
- Streamlit Cloud Main file path `showcase/Home.py`와 Python 3.11을 안내했는가
