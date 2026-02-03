@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements.txt

echo Cleaning up previous builds...
rmdir /s /q build
rmdir /s /q dist

echo Building executable...
python -m PyInstaller ThaiDanceScoring.spec

echo Done! The executable is in the dist folder.
pause
