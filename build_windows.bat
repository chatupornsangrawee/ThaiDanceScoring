@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Cleaning up previous builds...
rmdir /s /q build
rmdir /s /q dist

echo Building executable...
pyinstaller ThaiDanceScoring.spec

echo Done! The executable is in the dist folder.
pause
