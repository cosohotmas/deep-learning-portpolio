"""두 미니프로젝트용 showcase 디자인 헬퍼.

Part I showcase의 `Home.py → 프로젝트 카드 → 세부 페이지` 구조를 계승하되,
이 과제에서는 모델 학습 과정을 설명하는 '실험 노트 / 모델 여권' 콘셉트로 단순화합니다.
"""

import html

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#17324d; --paper:#f7f1e5; --coral:#e76f51; --mint:#2a9d8f; --line:#d8c8ad; }
        .stApp { background:linear-gradient(135deg,#fbf8f1 0%,#eef6f3 100%); color:var(--ink); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#fffaf0; border-right:1px solid var(--line); }
        .sc-hero { position:relative; overflow:hidden; padding:2.4rem; border:1px solid var(--line);
          border-radius:22px; background:rgba(255,255,255,.82); box-shadow:0 16px 36px rgba(23,50,77,.09); }
        .sc-hero:after { content:""; position:absolute; width:240px; height:240px; right:-90px; top:-100px;
          border-radius:50%; background:radial-gradient(circle,var(--coral),transparent 70%); opacity:.22; }
        .sc-kicker { color:var(--coral); font-weight:800; letter-spacing:.1em; font-size:.8rem; }
        .sc-title { color:var(--ink); font-size:clamp(2.1rem,5vw,4rem); line-height:1.02; margin:.5rem 0 .8rem; }
        .sc-sub { color:#506579; font-size:1.05rem; max-width:760px; margin:0; }
        .sc-chip { display:inline-block; margin:.35rem .35rem .35rem 0; padding:.35rem .65rem;
          border-radius:999px; background:#edf7f4; border:1px solid #bbd9d1; color:#1c6157; font-size:.82rem; }
        .sc-note { padding:1rem 1.1rem; border-left:4px solid var(--mint); background:rgba(255,255,255,.65); border-radius:0 12px 12px 0; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; background:rgba(255,255,255,.7); }
        [data-testid="stMetricValue"] { color:var(--ink); }
        .stButton>button { border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, name: str) -> None:
    st.markdown(
        f"""
        <section class="sc-hero">
          <div class="sc-kicker">DEEP LEARNING · TWO MODEL STORIES</div>
          <h1 class="sc-title">{html.escape(title)}</h1>
          <p class="sc-sub">{html.escape(subtitle)}</p>
          <div style="margin-top:1rem"><span class="sc-chip">Classification</span><span class="sc-chip">Regression</span><span class="sc-chip">PyTorch</span><span class="sc-chip">Streamlit</span></div>
          <p style="margin:.8rem 0 0;color:#6b7d8c">Built by {html.escape(name)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
