"""학생 앱을 showcase 페이지로 불러오는 작은 연결 계층."""

import importlib.util
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_student_app(relative_path: str, module_name: str) -> None:
    app_path = PROJECT_ROOT / relative_path
    if not app_path.exists():
        st.error(f"앱 파일을 찾을 수 없습니다: {relative_path}")
        st.stop()

    spec = importlib.util.spec_from_file_location(module_name, app_path)
    if spec is None or spec.loader is None:
        st.error(f"앱 모듈을 불러올 수 없습니다: {relative_path}")
        st.stop()
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.main()
