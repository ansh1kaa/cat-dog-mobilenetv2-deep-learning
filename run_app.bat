@echo off
REM Launch the Cat & Dog AI Classifier (run "pip install -r requirements.txt" once first)
cd /d "%~dp0"
python -m streamlit run app.py
pause
