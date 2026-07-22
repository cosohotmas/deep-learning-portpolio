# 서울 자전거 수요 예측기 — 학생용 핵심 스캐폴딩
# 실행: python3.11 -m streamlit run app_bike.py
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn as nn


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bike_reg.pt"
MY_NAME = "차백호"
REQUIRED_FEATURES = ["시간", "기온", "습도", "풍속", "가시거리", "이슬점", "일사량", "강우량", "적설량"]


class ScaffoldIncomplete(RuntimeError):
    """학생이 아직 완성하지 않은 핵심 구간을 화면에 친절하게 알립니다."""


def build_model(config: dict) -> nn.Module:
    """체크포인트 설정과 같은 회귀 신경망을 만듭니다."""
    # TODO 1 — 노트북 [Step 5]와 같은 Linear-ReLU 구조를 작성하세요.
    # config에는 input_dim과 hidden이 있습니다. 출력은 '대여량 하나'입니다.
    # 질문: 마지막 층에 ReLU를 두면 음수 예측을 막을 수 있지만 학습에는 어떤 영향이 있을까요?
    in_dim = config.get("input_dim", 9)
    hidden = config.get("hidden", 64)
    
    model = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1)  # 최종 대여량 연속값 딱 1개(대)를 출력합니다. (ReLU 없음!)
    )
    return model


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])

    # TODO 2 — state_dict를 복원하고 추론 모드로 전환하세요.
    # 반환 계약: (model, checkpoint)
    # TODO 2 완료 — state_dict를 뼈대에 복원하고 완벽하게 추론 모드(eval)로 전환합니다.
    
    # 1. 저장된 뇌세포 가중치 데이터를 뼈대에 완벽히 주입합니다.
    model.load_state_dict(checkpoint["state_dict"])
    
    # 2. 실실시간 웹 앱 추론 시 Dropout 등 학습용 가속 장치가 멋대로 가중치를 끄지 않도록 평가 모드로 전환합니다.
    model.eval()

    # 반환 계약: (model, checkpoint)
    return model, checkpoint


def prepare_features(values: dict[str, float], checkpoint: dict) -> torch.Tensor:
    """화면 입력을 학습 때와 같은 특성 순서·스케일로 변환합니다."""
    feature_names = list(checkpoint["feature_names"])

    # TODO 3 — 아래 계약을 만족하는 normalized tensor를 반환하세요.
    # ① raw는 checkpoint["mean"]에서 시작합니다. 화면에 없는 특성도 안전한 기본값을 갖게 됩니다.
    # ② values의 각 값을 feature_names가 정한 정확한 위치에 반영합니다.
    # ③ checkpoint의 train mean/std로 표준화합니다. 앱 데이터로 새 통계를 계산하면 안 됩니다.
    # 최종 shape: (특성 수,)
    # ① 학습 당시 저장해 둔 평균(mean) 값을 초기 기본값 배열로 사용하여 안전장치를 만듭니다.
    #    (화면에 없는 특성이 혹시 있더라도 결측치 에러가 안 나고 안전한 기본값을 갖게 됩니다.)
    raw_array = np.array(checkpoint["mean"], dtype="float32")
    
    # ② 사용자가 화면 슬라이더로 조정한 실시간 입력값(values)을 원래의 정확한 특성 위치(인덱스)에 반영합니다.
    for key, val in values.items():
        if key in feature_names:
            idx = feature_names.index(key)
            raw_array[idx] = val
            
    # ③ 과거 학습용 데이터의 평균(mean)과 표준편차(std) 자(Ruler)를 그대로 들이대어 표준화합니다.
    #    (주의: 절대로 앱 안에서 실시간 데이터로 새로운 통계를 계산하면 안 됩니다!)
    mean_ref = np.array(checkpoint["mean"], dtype="float32")
    std_ref = np.array(checkpoint["std"], dtype="float32")
    std_ref[std_ref == 0] = 1.0  # 0 나누기 방지 안전망
    
    normalized_array = (raw_array - mean_ref) / std_ref
    
    # 최종 반환 계약: (특성 수,) 크기를 가지는 파이토치 float32 텐서
    return torch.tensor(normalized_array, dtype=torch.float32)


def predict_count(model: nn.Module, normalized: torch.Tensor) -> float:
    """표준화된 한 행으로 시간당 대여량을 예측합니다."""
    # TODO 4 — gradient 기록 없이 배치 차원을 추가해 순전파하고 Python float으로 반환하세요.
    # 질문: 모델 출력 shape (1,1)에서 어느 값을 선택해야 하나요?
    # 1. 7강/13강 계약: 그래디언트 메모리 찌꺼기가 쌓이지 않도록 완벽하게 원천 차단합니다.
    with torch.inference_mode():
        # 2. 모델은 무조건 배치(묶음) 차원을 원하므로, 1차원 (9,) 텐서 앞에 
        #    unsqueeze(0)를 붙여 2차원 배치 형태인 (1, 9) 상자로 안전하게 포장하여 밀어 넣습니다.
        x_in = normalized.unsqueeze(0)
        
        # 3. 모델에 입력하여 미래의 자전거 대여 대수 점수(Shape: 1, 1)를 출력합니다.
        pred_tensor = model(x_in)
        
    # 4. 출력용 텐서 상자 (1, 1) 안에 고독하게 갇혀있는 단 하나의 예측 실수값을 
    #    .item() 메서드로 쏙 뽑아내어 순수한 Python float 스칼라 형태로 반환합니다.
    return float(pred_tensor.item())


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#17324d; --paper:#f7f1e5; --coral:#e76f51; --mint:#2a9d8f; }
        .stApp { background: linear-gradient(135deg, #fbf8f1 0%, #eef6f3 100%); color:var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background:#fffaf0; border-right:1px solid #decfb8; }
        .mp-hero { padding:1.6rem 1.8rem; border:1px solid #d8c8ad; border-radius:18px;
          background:rgba(255,255,255,.82); box-shadow:0 12px 30px rgba(23,50,77,.08); margin-bottom:1.2rem; }
        .mp-kicker { color:var(--mint); font-weight:800; letter-spacing:.08em; font-size:.78rem; }
        .mp-title { color:var(--ink); font-size:clamp(1.8rem,4vw,3rem); line-height:1.08; margin:.35rem 0; }
        .mp-sub { color:#506579; margin:0; max-width:780px; }
        .mp-step { border-left:4px solid var(--coral); padding:.35rem .8rem; color:#40566b; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; background:rgba(255,255,255,.68); }
        .stButton>button { border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="mp-hero">
          <div class="mp-kicker">MODEL LAB · REGRESSION</div>
          <h1 class="mp-title">날씨와 시간이<br>대여량 하나가 되기까지</h1>
          <p class="mp-sub">학습 때 저장한 특성 순서와 train 통계를 그대로 복원해 미래 시점의 수요를 예측합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="자전거 회귀 모델 랩", page_icon="🚲", layout="wide")
    apply_page_style()
    render_header()

    with st.expander("🧭 이 앱에서 내가 직접 완성할 핵심 4단계", expanded=False):
        st.markdown(
            """
            1. `build_model()` — 체크포인트 설정으로 회귀 구조 재구성
            2. `load_model()` — `state_dict` 복원과 추론 모드 전환
            3. `prepare_features()` — 특성 순서 복원과 train 통계 표준화
            4. `predict_count()` — 한 행을 배치로 바꿔 예측값 반환

            슬라이더와 결과 패널은 제공됩니다. 네 함수가 연결되어야 실제 예측이 시작됩니다.
            """
        )

    if not MODEL_PATH.exists():
        st.error("`bike_reg.pt`가 없습니다. 과제 노트북의 체크포인트 저장 셀을 실행하세요.")
        st.stop()

    try:
        model, checkpoint = load_model()
    except ScaffoldIncomplete as exc:
        st.warning(str(exc))
        st.info("`app_bike.py`의 TODO 1→2 순서로 완성한 뒤 파일을 저장하면 화면이 자동으로 다시 실행됩니다.")
        st.stop()
    except (KeyError, RuntimeError, TypeError) as exc:
        st.error("학습 때의 모델 구조와 앱의 구조가 일치하지 않습니다. TODO 1·2와 checkpoint key를 확인하세요.")
        st.code(str(exc))
        st.stop()

    feature_names = list(checkpoint["feature_names"])
    missing = [name for name in REQUIRED_FEATURES if name not in feature_names]
    if missing:
        st.error(f"필수 특성이 체크포인트에 없습니다: {missing}")
        st.stop()

    metrics = checkpoint["metrics"]
    train_config = checkpoint["training_config"]
    model_config = checkpoint["model_config"]
    st.sidebar.header("MODEL PASSPORT")
    st.sidebar.metric("Validation MAE", f"{metrics['val_mae']:,.1f}대")
    st.sidebar.metric("최종 Test MAE", f"{metrics['test_mae']:,.1f}대")
    st.sidebar.caption(f"hidden {model_config['hidden']} · epochs {train_config['epochs']}")
    st.sidebar.caption(f"lr {train_config['lr']} · batch {train_config['batch']}")
    st.sidebar.caption(f"제작: {MY_NAME}")

    input_col, result_col = st.columns([1.35, 0.65], gap="large")
    with input_col:
        st.subheader("01 · 예측 조건")
        row1 = st.columns(3)
        hour = row1[0].slider("시간", 0, 23, 8)
        temperature = row1[1].slider("기온(°C)", -18.0, 40.0, 20.0, 0.5)
        humidity = row1[2].slider("습도(%)", 0, 100, 55)
        row2 = st.columns(3)
        wind = row2[0].slider("풍속(m/s)", 0.0, 8.0, 1.5, 0.1)
        visibility = row2[1].slider("가시거리(10m)", 0, 2000, 1500, 50)
        dew_point = row2[2].slider("이슬점(°C)", -30.0, 28.0, 10.0, 0.5)
        with st.expander("강수·일사 조건", expanded=True):
            row3 = st.columns(3)
            solar = row3[0].slider("일사량(MJ/m²)", 0.0, 3.6, 0.5, 0.1)
            rain = row3[1].slider("강우량(mm)", 0.0, 35.0, 0.0, 0.1)
            snow = row3[2].slider("적설량(cm)", 0.0, 9.0, 0.0, 0.1)

    values = {
        "시간": hour, "기온": temperature, "습도": humidity, "풍속": wind,
        "가시거리": visibility, "이슬점": dew_point, "일사량": solar,
        "강우량": rain, "적설량": snow,
    }
    with result_col:
        st.subheader("02 · 예측 결과")
        try:
            normalized = prepare_features(values, checkpoint)
            prediction = predict_count(model, normalized)
        except ScaffoldIncomplete as exc:
            st.warning(str(exc))
            st.stop()
        with st.container(border=True):
            st.metric("예상 대여량", f"{max(0, prediction):,.0f}대/시간")
            st.caption(f"Validation MAE 기준, 평균적으로 약 {metrics['val_mae']:,.0f}대 차이가 날 수 있습니다.")
        st.markdown('<p class="mp-step">슬라이더 하나만 바꿔 예측이 어떻게 움직이는지 관찰하고, 데이터에서 배운 관계인지 설명해 보세요.</p>', unsafe_allow_html=True)
        st.caption("2017-12~2018-11 운영일 자료로 학습한 교육용 모델입니다. 실제 운영에는 최신 데이터와 불확실성 검증이 필요합니다.")


if __name__ == "__main__":
    main()
