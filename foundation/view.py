"""Watch a saved foundation ONNX actor using inference only (no Torch required)."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path

from .onnx_policy import Policy as OnnxPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'data/neon_foundation_round03/foundation.onnx'


class LocalPolicy(OnnxPolicy):
    def __init__(self, player_id=0, config=None):
        model = Path(os.environ.get('NEON_VIEW_MODEL', str(DEFAULT_MODEL)))
        super().__init__(player_id, config, model_path=model,
                         metadata_path=model.with_name('metadata.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--seed', type=int, default=19400)
    parser.add_argument('--scene', choices=('room', 'cyber_city'), default='room')
    parser.add_argument('--headless', action='store_true', help='Run and save a replay without opening a window.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    if not model.is_file() or not model.with_name('metadata.json').is_file():
        parser.error('Both the ONNX model and its sibling metadata.json are required')
    if args.duration <= 0 or args.seed < 0:
        parser.error('duration must be positive and seed nonnegative')
    output = (args.output or ROOT / 'artifacts/foundation_view' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be new or empty to preserve existing replays')
    os.environ['NEON_VIEW_MODEL'] = str(model)
    print(f'Inference only: ONNX Runtime CPU; no gradients or optimizer. Model: {model}', flush=True)
    from embodied_arena.cli import run
    run(argparse.Namespace(
        config=str(ROOT / ('configs/room.toml' if args.scene == 'room' else 'configs/default.toml')),
        duration=args.duration, seed=args.seed, scene=args.scene,
        output=str(output), policy_a='foundation.view:LocalPolicy',
        policy_b='embodied_arena.policies:WalkInPlace',
        camera=None, inline=True, viewer=not args.headless, realtime=not args.headless,
        render_mode='simple', no_cameras=False, no_preview=args.headless, gif=False, timeout=1.,
    ))


if __name__ == '__main__':
    main()
