# MNIST 손글씨 분류기 — 학생용 핵심 스캐폴딩
# 실행: python3.11 -m streamlit run app.py
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "mnist_cnn.pt"
MY_NAME = "차백호"


class ScaffoldIncomplete(RuntimeError):
    """학생이 아직 완성하지 않은 핵심 구간을 화면에 친절하게 알립니다."""


class CNN(nn.Module):
    def __init__(self, conv1=32, conv2=64, hidden=128, dropout=0.3):
        super().__init__()

        # TODO 1-A — 노트북 [Step 3]의 CNN과 같은 feature extractor를 작성하세요.
        # 확인할 shape 흐름:
        # (N, 1, 28, 28) → 첫 합성곱/풀링 → (N, conv1, 14, 14)
        #                    → 둘째 합성곱/풀링 → (N, conv2, 7, 7)
        # 질문: ReLU는 어느 연산 뒤에 있어야 하고, MaxPool2d는 공간 크기를 어떻게 바꾸나요?
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                     # (N, conv1, 14, 14)
            nn.Conv2d(conv1, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                     # (N, conv2, 7, 7)
        )

        # TODO 1-B — 7×7 feature map을 숫자 10개의 logit으로 바꾸는 분류부를 작성하세요.
        # 체크포인트의 conv2·hidden 값이 달라도 동작해야 하므로 숫자를 고정하지 마세요.
        # 마지막 층에 Softmax를 작성하지 않습니다. 학습한 모델은 logit을 출력합니다.
        self.classifier = nn.Sequential(
            nn.Flatten(),                                        # (N, conv2 * 7 * 7)
            nn.Linear(conv2 * 7 * 7, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),                               # 최종 클래스 10개 로짓 (Softmax 없음)
        )

    def forward(self, x):
        # TODO 1-C — features의 출력을 classifier로 전달해 반환하세요.
        # 먼저 중간 shape을 print해 보면 두 블록이 왜 나뉘는지 이해하기 쉽습니다.
        
        # 1. 이미지를 특징 추출기(features)에 넣어 7x7 특징 맵으로 만듭니다.
        feat = self.features(x)
        
        # 2. 특징 맵을 분류기(classifier)에 넣어 최종 10개 숫자의 점수(로짓)를 뽑아 반환합니다.
        return self.classifier(feat)


@st.cache_resource
def load_model():
    """체크포인트로 학습 때와 같은 CNN을 복원합니다."""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    # TODO 2 — 아래 세 단계를 직접 작성하세요.
    # ① checkpoint["model_config"] 딕셔너리를 언패킹(**)하여 학습 당시의 손잡이 크기 그대로 모델 인스턴스를 만듭니다.
    cfg = checkpoint["model_config"]
    model = CNN(
        conv1=cfg["conv1"], 
        conv2=cfg["conv2"], 
        hidden=cfg["hidden"], 
        dropout=cfg["dropout"]
    )
    
    # ② 저장된 뇌세포(무게중심) 데이터인 state_dict를 뼈대에 이식합니다.
    model.load_state_dict(checkpoint["state_dict"])
    
    # ③ 테스트 및 추론할 때는 Dropout 같은 학습용 장치를 꺼야 하므로 반드시 eval() 모드로 전환합니다.
    model.eval()

    # 반환 계약: (model, checkpoint)
    return model, checkpoint


def make_mnist_canvas(image: Image.Image) -> np.ndarray:
    """화면용 보조 코드: 다양한 사진을 MNIST와 비슷한 28×28 캔버스로 정렬합니다."""
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    gray = ImageOps.grayscale(Image.alpha_composite(white, rgba).convert("RGB"))
    arr = np.array(gray, dtype=np.uint8)

    if float(arr.mean()) > 127:
        arr = 255 - arr
    arr[arr < 30] = 0

    mask = arr > 0
    if mask.any():
        ys, xs = np.where(mask)
        digit = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        digit = arr

    height, width = digit.shape
    side = max(height, width, 1)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = digit

    resized = Image.fromarray(square).resize((20, 20), Image.Resampling.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = np.asarray(resized, dtype=np.uint8)
    return canvas


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """업로드 이미지 → 모델 입력 tensor와 화면용 28×28 배열."""
    canvas = make_mnist_canvas(image)

    # TODO 3 — canvas를 학습 때와 같은 입력 계약으로 변환하세요.
    # 최종 확인: shape (1, 1, 28, 28), dtype float32, 값 범위 0~1
    # 첫 번째 1은 배치, 두 번째 1은 흑백 채널입니다.
    # TODO 3 완료 — canvas(28x28)를 (1, 1, 28, 28) 형태의 float32 텐서로 변환하고 0~1 정규화를 마쳤습니다.
    
    # 1. 0~255 정수형 데이터를 0.0~1.0 실수형 데이터로 정규화합니다.
    img_scaled = canvas.astype("float32") / 255.0
    
    # 2. 파이토치가 먹을 수 있게 텐서로 변환하고 채널과 배치 축을 확장합니다 (28, 28) -> (1, 1, 28, 28)
    tensor_in = torch.tensor(img_scaled).unsqueeze(0).unsqueeze(0)

    # 반환 계약: (모델 입력용 4D 텐서, 화면 출력용 2D 배열)
    return tensor_in, canvas


def predict_probabilities(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """모델 logit을 숫자 0~9의 확률 배열로 변환합니다."""
    # TODO 4 — gradient 기록 없이 순전파하고, 클래스 축에 softmax를 적용하세요.
    # 반환 계약: shape (10,)의 NumPy 배열
    # 질문: argmax만 사용하면 무엇을 잃고, softmax를 두 번 적용하면 왜 잘못될까요?
    """모델 logit을 숫자 0~9의 확률 배열로 변환합니다."""
    # TODO 4 완료 — inference_mode로 연산 찌꺼기를 막고, 순전파 후 softmax를 적용해 NumPy 배열로 반환합니다.
    
    # 1. 7강/13강 계약: 평가 및 추론 시 불필요한 그래디언트 메모리 기록을 완벽히 차단합니다.
    with torch.inference_mode():
        # 2. 모델에 4차원 텐서 x를 밀어 넣어 10개 클래스의 로짓(점수)을 얻습니다. (Shape: 1, 10)
        logits = model(x)
        
        # 3. 로짓 점수를 0~1 사이의 확률값으로 변환합니다. (클래스 축인 dim=1 기준)
        probabilities = torch.softmax(logits, dim=1)
    
    # 4. 화면 UI 출력을 위해 텐서를 CPU로 내리고 연산 그래프를 끊은 뒤, 1차원 (10,) 규격의 NumPy 배열로 풀어줍니다.
    return probabilities.squeeze(0).cpu().numpy()


def apply_page_style() -> None:
    """Part I showcase의 카드형 레이아웃을 이 과제의 실험 노트 콘셉트로 축소했습니다."""
    st.markdown(
        """
        <style>
        :root { --ink:#17324d; --paper:#f7f1e5; --coral:#e76f51; --mint:#2a9d8f; }
        .stApp { background: linear-gradient(135deg, #fbf8f1 0%, #eef6f3 100%); color:var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background:#fffaf0; border-right:1px solid #decfb8; }
        .mp-hero { padding:1.6rem 1.8rem; border:1px solid #d8c8ad; border-radius:18px;
          background:rgba(255,255,255,.82); box-shadow:0 12px 30px rgba(23,50,77,.08); margin-bottom:1.2rem; }
        .mp-kicker { color:var(--coral); font-weight:800; letter-spacing:.08em; font-size:.78rem; }
        .mp-title { color:var(--ink); font-size:clamp(1.8rem,4vw,3rem); line-height:1.08; margin:.35rem 0; }
        .mp-sub { color:#506579; margin:0; max-width:760px; }
        .mp-step { border-left:4px solid var(--mint); padding:.35rem .8rem; color:#40566b; }
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
          <div class="mp-kicker">MODEL LAB · CLASSIFICATION</div>
          <h1 class="mp-title">손글씨 한 장이<br>열 개의 확률이 되기까지</h1>
          <p class="mp-sub">직접 학습한 CNN의 체크포인트를 복원하고, 사진을 28×28 tensor로 바꿔 예측합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="MNIST 분류 모델 랩", page_icon="✍️", layout="wide")
    apply_page_style()
    render_header()

    with st.expander("🧭 이 앱에서 내가 직접 완성할 핵심 4단계", expanded=False):
        st.markdown(
            """
            1. `CNN` — 학습 때와 같은 모델 구조 재구성
            2. `load_model()` — 구조와 `state_dict` 복원, 추론 모드 전환
            3. `preprocess_image()` — 사진을 `(1,1,28,28)` 입력으로 변환
            4. `predict_probabilities()` — logit을 10개 확률로 변환

            화면 구성은 제공됩니다. 네 함수가 연결되어야 실제 예측이 시작됩니다.
            """
        )

    if not MODEL_PATH.exists():
        st.error("`mnist_cnn.pt`가 없습니다. 과제 노트북 [Step 7]을 실행해 이 폴더에 생성하세요.")
        st.stop()

    try:
        model, checkpoint = load_model()
    except ScaffoldIncomplete as exc:
        st.warning(str(exc))
        st.info("`app.py`의 TODO 1→2 순서로 완성한 뒤 파일을 저장하면 화면이 자동으로 다시 실행됩니다.")
        st.stop()
    except (KeyError, RuntimeError, TypeError) as exc:
        st.error("학습 때의 모델 구조와 앱의 구조가 일치하지 않습니다. TODO 1·2와 checkpoint key를 확인하세요.")
        st.code(str(exc))
        st.stop()

    metrics = checkpoint["metrics"]
    train_config = checkpoint["training_config"]
    model_config = checkpoint["model_config"]
    st.sidebar.header("MODEL PASSPORT")
    st.sidebar.metric("Validation 정확도", f"{metrics['val_acc']:.4f}")
    st.sidebar.metric("최종 Test 정확도", f"{metrics['test_acc']:.4f}")
    st.sidebar.metric("파라미터", f"{checkpoint['n_params']:,}")
    st.sidebar.caption(f"epochs {train_config['epochs']} · lr {train_config['lr']}")
    st.sidebar.caption(f"conv {model_config['conv1']}/{model_config['conv2']} · hidden {model_config['hidden']}")
    st.sidebar.caption(f"제작: {MY_NAME}")

    input_col, result_col = st.columns([1, 1], gap="large")
    image = x = preview = None
    with input_col:
        st.subheader("01 · 입력 이미지")
        method = st.radio("입력 방식", ["이미지 업로드", "카메라 촬영"], horizontal=True)
        if method == "이미지 업로드":
            source = st.file_uploader("PNG 또는 JPG", type=["png", "jpg", "jpeg"])
        else:
            source = st.camera_input("종이의 숫자를 크게 촬영하세요")

        if source is None:
            st.info("숫자 한 개가 크게 보이는 이미지를 준비하세요. 흰 배경·검은 글씨도 자동으로 반전합니다.")
            st.markdown('<p class="mp-step">관찰 포인트: 학습 데이터와 직접 찍은 사진의 분포는 어떻게 다를까요?</p>', unsafe_allow_html=True)
        else:
            try:
                image = Image.open(source)
                x, preview = preprocess_image(image)
            except ScaffoldIncomplete as exc:
                st.warning(str(exc))
                st.stop()
            except Exception:
                st.error("이미지를 열 수 없습니다. 손상되지 않은 PNG/JPG 파일인지 확인하세요.")
                st.stop()

            before, after = st.columns(2)
            before.image(image, caption="원본", width="stretch")
            after.image(preview, caption="모델 입력 28×28", clamp=True, width="stretch")

    with result_col:
        st.subheader("02 · 모델의 판단")
        if source is None:
            with st.container(border=True):
                st.markdown("### 결과 대기 중")
                st.caption("왼쪽에서 이미지를 선택하면 예측 숫자와 클래스 확률이 여기에 표시됩니다.")
            return
        try:
            probabilities = predict_probabilities(model, x)
        except ScaffoldIncomplete as exc:
            st.warning(str(exc))
            st.stop()
        prediction = int(probabilities.argmax())
        with st.container(border=True):
            st.metric("예측 숫자", prediction)
            st.caption(f"가장 높은 softmax 점수 · {probabilities[prediction] * 100:.1f}%")
            st.bar_chart({"클래스 확률": probabilities})
            top3 = probabilities.argsort()[-3:][::-1]
            st.caption("Top 3 · " + " · ".join(f"{int(i)}: {probabilities[i]:.1%}" for i in top3))
        st.caption("Test 정확도는 정제된 데이터의 성적입니다. 촬영 사진은 배경·크기·중심이 달라 같은 성능을 보장하지 않습니다.")


if __name__ == "__main__":
    main()
