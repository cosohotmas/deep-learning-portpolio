import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from showcase.core.loader import run_student_app


run_student_app("과제_Streamlit_앱_회귀/app_bike.py", "student_regression_app")
