@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -m embodied_arena run --config configs\room.toml --duration 20 --viewer --realtime --output artifacts\room_latest
pause
