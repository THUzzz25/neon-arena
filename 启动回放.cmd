@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "ARENA_REPLAY=artifacts\latest"
if not exist "%ARENA_REPLAY%\trajectory.npz" set "ARENA_REPLAY=artifacts\cyber_city_match"
if not exist "%ARENA_REPLAY%\trajectory.npz" set "ARENA_REPLAY=artifacts\demo"
".venv\Scripts\python.exe" -m embodied_arena replay "%ARENA_REPLAY%"
pause
