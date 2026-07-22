import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from showcase.core.theme import apply_theme, hero
from showcase.profile import PROFILE


st.set_page_config(page_title="나의 딥러닝 모델 랩", page_icon="🧪", layout="wide")
apply_theme()
hero(PROFILE["title"], PROFILE["subtitle"], PROFILE["name"])

classification_model = PROJECT_ROOT / "과제_Streamlit_앱_분류/mnist_cnn.pt"
regression_model = PROJECT_ROOT / "과제_Streamlit_앱_회귀/bike_reg.pt"
ready_count = sum(path.exists() for path in [classification_model, regression_model])

st.markdown("### 프로젝트 현황")
m1, m2, m3 = st.columns(3)
m1.metric("완성한 모델", f"{ready_count}/2")
m2.metric("문제 유형", "분류 + 회귀")
m3.metric("최종 데모", "1 showcase")

left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True):
        st.markdown("### ✍️ 이미지 분류")
        st.caption("MNIST · CNN · Accuracy")
        st.write("사진을 28×28 tensor로 변환하고, 학습한 CNN이 숫자 0~9의 확률을 출력합니다.")
        st.success("체크포인트 준비 완료" if classification_model.exists() else "노트북 [Step 7] 완료 후 체크포인트 생성")
        st.page_link("pages/1_분류_MNIST.py", label="분류 모델 열기 →", use_container_width=True)
with right:
    with st.container(border=True):
        st.markdown("### 🚲 수요 회귀")
        st.caption("Seoul Bike · MLP · MAE")
        st.write("시간과 날씨를 train 통계로 표준화하고, 미래 시점의 시간당 대여량을 예측합니다.")
        st.success("체크포인트 준비 완료" if regression_model.exists() else "체크포인트 저장 셀 완료 후 생성")
        st.page_link("pages/2_회귀_자전거.py", label="회귀 모델 열기 →", use_container_width=True)

st.markdown("### 내가 설명할 수 있어야 하는 것")
q1, q2, q3 = st.columns(3)
with q1:
    with st.container(border=True):
        st.markdown("**01 · 분류의 한계**")
        st.write(PROFILE["classification_insight"])
with q2:
    with st.container(border=True):
        st.markdown("**02 · 회귀의 오차**")
        st.write(PROFILE["regression_insight"])
with q3:
    with st.container(border=True):
        st.markdown("**03 · 다음 실험**")
        st.write(PROFILE["next_step"])

with st.expander("🧩 두 프로젝트가 하나의 showcase가 되는 구조"):
    st.code(
        """노트북 1 ──학습/저장──> mnist_cnn.pt ──> 분류 앱 ─┐
                                                        ├─> showcase/Home.py
노트북 2 ──학습/저장──> bike_reg.pt  ──> 회귀 앱 ────┘""",
        language="text",
    )
    st.markdown(
        '<div class="sc-note">Part I showcase와 같은 단방향 구조입니다. showcase는 두 앱을 재사용하지만, 각 앱은 showcase 없이도 독립 실행됩니다.</div>',
        unsafe_allow_html=True,
    )
