@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -m embodied_arena.platform.deploy
if errorlevel 1 (
  echo 构建失败。请确认 Docker Desktop 已启动，并可连接官方镜像下载源。
) else (
  echo 统一运行环境已构建，请在管理员页面点击重新检测。
)
pause
