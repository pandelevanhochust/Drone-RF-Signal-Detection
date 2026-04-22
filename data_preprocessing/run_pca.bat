@echo off
echo Installing required libraries...
pip install -r requirements.txt
echo.
echo Running pca.py...
python pca.py
pause