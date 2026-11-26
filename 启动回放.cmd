@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "ARENA_REPLAY=artifacts\latest"
if not exist "%ARENA_REPLAY%\trajectory.npz" set "ARENA_REPLAY=artifacts\cyber_city_match"
if not exist "%ARENA_REPLAY%\trajectory.npz" set "ARENA_REPLAY=artifacts\demo"
echo 精简渲染模式：点击窗口按钮切换选手视角、全局视角和跟随视角。
".venv\Scripts\python.exe" -m embodied_arena replay "%ARENA_REPLAY%" --render-mode simple
pause
