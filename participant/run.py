"""Python-only local participant entry point, with no account or container service."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="THUAI10 本地选手包：修改 policy.py，运行仿真，查看结果")
    parser.add_argument("--scene", choices=["room", "cyber_city"], default="room")
    parser.add_argument("--duration", type=positive, default=5.0, help="仿真秒数，默认 5 秒")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", default="policy:Policy", help="自己的 模块:类名")
    parser.add_argument("--opponent", default="policy:Policy", help="对手 模块:类名（独立实例）")
    parser.add_argument("--viewer", action="store_true", help="打开实时仿真窗口")
    parser.add_argument("--render-mode", choices=["simple", "debug"], default="simple",
                        help="窗口模式：simple 精简视角按钮（默认），debug 完整 MuJoCo 调试面板")
    parser.add_argument("--no-cameras", action="store_true", help="关闭 RGB-D 和预览，仅检查物理链路")
    parser.add_argument("--output", type=Path, help="结果目录，必须不存在或为空")
    parser.add_argument("--replay", type=Path, help="回放已保存的结果目录")
    parser.add_argument("--speed", type=positive, default=1.0, help="回放倍速")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    from embodied_arena import cli

    if args.replay:
        cli.replay(argparse.Namespace(directory=args.replay.resolve(), speed=args.speed,
                                      camera=None, render_mode=args.render_mode))
        return 0

    output = (args.output or ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"结果目录必须为空，避免覆盖已有回放：{output}")
    print("Running local Python policies (no login / Docker).", flush=True)
    cli.run(argparse.Namespace(
        config=None, duration=args.duration, seed=args.seed, scene=args.scene,
        output=str(output), policy_a=args.policy, policy_b=args.opponent,
        camera=None, inline=True, viewer=args.viewer, realtime=args.viewer, render_mode=args.render_mode,
        no_cameras=args.no_cameras, no_preview=args.no_cameras, gif=False,
        timeout=1.0,
    ))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if result["events"].get("invalid_action", 0):
        print(f"策略返回了无效动作，详情见 {output / 'events.jsonl'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
