@echo off
echo Installing required libraries...
pip install -r requirement.txt
echo.
echo Running transform.py...
python transform.py
pause