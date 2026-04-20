@echo off
echo Installing required libraries...
pip install -r requirements.txt
echo.
echo Running transform.py...
python transform.py
pause