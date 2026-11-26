@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行安装命令：powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  pause
  exit /b 1
)
echo 精简渲染模式：点击窗口按钮切换选手 1、选手 2、全局、跟随选手 1 或跟随选手 2。
echo 如需 MuJoCo 调试面板，请在命令行使用 --render-mode debug。
".venv\Scripts\python.exe" -m embodied_arena run --scene cyber_city --duration 60 --viewer --realtime --render-mode simple --output artifacts\latest
pause
