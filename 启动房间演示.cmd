@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 精简渲染模式：点击窗口按钮切换选手视角、全局视角和跟随视角。
".venv\Scripts\python.exe" -m embodied_arena run --config configs\room.toml --duration 20 --viewer --realtime --render-mode simple --output artifacts\room_latest
pause
