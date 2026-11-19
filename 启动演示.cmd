@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行安装命令：powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  pause
  exit /b 1
)
echo 赛博街区镜头：0 总览，1 街道，2 教堂，3 商铺，4 充电区，5 公园，6 咖啡厅，7 档案馆，8 地图，9 机器人视角。
echo 新增镜头：N 夜市街口，D 空中无人机，P 树木，W 西侧街区，H 天际线；S 驿站，T 南街蜜雪，F 夜市内部，G 机厅。
echo 双街镜头：0 全景，U 生活商业街，K 咖啡小吃店，L 酒店，J 连接路口，B 原霓虹主街。
".venv\Scripts\python.exe" -m embodied_arena run --scene cyber_city --duration 60 --viewer --realtime --output artifacts\latest
pause
