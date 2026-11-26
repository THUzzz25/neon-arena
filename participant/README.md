# THUAI10 最小 Python 选手包

这是独立可复制的本地选手包，内含原项目的真实 MuJoCo 仿真核心、G1 模型及城市资源，无需相邻的 `neon-arena` 目录。链路为 **修改 Python 策略 → 读取观测 → 返回动作 → 仿真与裁判 → 成绩和回放**。

这一版没有账号、权限管理、网页服务或 Docker。全部开发和启动操作使用 Python；物理引擎使用 pip 安装的 MuJoCo 二进制依赖，不需要编写或编译 C++。安装需要联网，安装完成后可离线运行。推荐 64 位 Python 3.13，最低 3.11；RGB-D 和窗口需要可用的 OpenGL 图形驱动。

## 三步运行

在 `participant` 目录打开终端：

```powershell
python setup_env.py
.venv/Scripts/python.exe run.py
.venv/Scripts/python.exe run.py --viewer
```

Linux/macOS 将 `.venv/Scripts/python.exe` 换成 `.venv/bin/python`。也可以用已有 Python 环境执行 `python -m pip install -r requirements.txt`，然后 `python run.py`。

默认运行 5 仿真秒的小房间，两个独立 `Policy` 实例对局，开启真实 RGB-D，不弹窗口。终端打印分数、生命值与结果目录。默认策略只保持初始关节目标，是可运行接口示例，不保证站立、行走或得分；零分平局是正常结果。

## 修改策略

只需编辑 `policy.py` 中的 `Policy.act(observation)`。构造参数为 `player_id=0, config=None`。

| 输入 | 内容 |
| --- | --- |
| `time`, `player`, `reset_count` | 仿真秒数、玩家编号、复位次数 |
| `proprio.joint_names` | 29 个关节名及动作顺序 |
| `proprio.q`, `proprio.dq` | 关节角 rad、角速度 rad/s |
| `proprio.base_quat`, `proprio.gyro` | wxyz 四元数、本体角速度 |
| `visual` | RGB-D；禁用相机时为 `None` |
| `visual.rgb`, `visual.depth` | RGB uint8 H×W×3、深度 H×W，单位米 |
| `visual.time` | 图像时间戳；图像 10 Hz，控制 50 Hz |
| `tasks` | 当前可见任务，含文字和 `reference_rgb` 参考图 |
| `game` | 自己的血量、得分、存活、冷却等状态 |

输出 `{"joint_targets": 29个有限实数的列表或数组, "fire": False}`。关节目标单位 rad，按 `joint_names` 排列；超限目标由引擎限幅，`fire` 必须为布尔值。复位次数变化时应清空策略的历史状态。无效动作记入事件文件，程序结束时返回非零退出码；Python 异常直接显示堆栈，便于调试。

策略在本地同一进程运行，无权限隔离和超时终止，只适合运行自己信任的代码。

## 练习、结果与回放

```powershell
# 更长的房间练习
.venv/Scripts/python.exe run.py --duration 30 --output outputs/my-run
# 城市地图（加载和渲染比房间慢）
.venv/Scripts/python.exe run.py --scene cyber_city --duration 5 --viewer
# 自定义对手：在包内创建 opponent.py，其中定义 Policy
.venv/Scripts/python.exe run.py --opponent opponent:Policy
# 无图形环境先检查物理链路（此时无 RGB-D 或预览）
.venv/Scripts/python.exe run.py --no-cameras
# 回放：替换为终端实际打印的结果路径
.venv/Scripts/python.exe run.py --replay outputs/my-run
```

默认结果保存在 `outputs/时间戳/`，显式结果目录必须为空，防止覆盖。

- `result.json`：得分、胜者、血量与事件计数。
- `preview.png`：对局结束时的场景与双方视觉预览（启用相机时）。
- `events.jsonl`：裁判事件，包括射击、复位和无效动作。
- `trajectory.npz`、`scene.xml`：真实状态轨迹与场景，用于回放。
- `config.json`、`environment.json`、`submissions.json`：配置、环境与策略入口。

`embodied_arena/` 是从现有项目提取的仿真核心快照，`configs/` 为本体和规则参数，`assets/`、`third_party/` 为随包资源。当前不包含上传、训练或预训练行走模型。第三方资源来源与许可证见 `THIRD_PARTY.md`。
