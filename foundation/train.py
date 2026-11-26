"""GPU behavior cloning of public-observation teacher/DAgger episodes.

Validation is split by complete episode, never by adjacent control frames.
Offline loss is diagnostic; promotion requires separate closed-loop evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .model import FoundationNet, save_checkpoint, validate_gait_import


class Episodes:
    def __init__(self, files):
        self.files = list(files)
        self.episodes = []
        for file in self.files:
            with np.load(file, allow_pickle=False) as data:
                self.episodes.append({key: data[key] for key in data.files if key != 'metadata'})
        self.lengths = np.asarray([len(x['action']) for x in self.episodes])
        self.ends = self.lengths.cumsum()
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError('No training observations')
        offsets = np.concatenate([[0], self.ends[:-1]])
        self.positive_indices = np.concatenate([np.flatnonzero(ep['fire'] > .5) + offset
                                               for ep, offset in zip(self.episodes, offsets)])

    def batch(self, size, rng, device, positive_fraction=0.):
        indices = rng.integers(0, self.ends[-1], size=size)
        if len(self.positive_indices) and positive_fraction > 0:
            count = round(size * positive_fraction)
            indices[:count] = rng.choice(self.positive_indices, count, replace=True)
        episodes = np.searchsorted(self.ends, indices, side='right')
        previous = np.concatenate([[0], self.ends[:-1]])
        rows = indices - previous[episodes]
        result = {key: [] for key in ('proprio', 'game', 'image', 'tasks', 'task_mask', 'action', 'command', 'fire', 'wrist_goal', 'wrist_goal_mask')}
        for ep, row in zip(episodes, rows):
            data = self.episodes[ep]
            for key in ('proprio', 'game', 'action', 'command', 'fire'):
                result[key].append(data[key][row])
            result['wrist_goal'].append(data['wrist_goal'][row] if 'wrist_goal' in data else data['action'][row, -2:])
            result['wrist_goal_mask'].append(float('wrist_goal' in data))
            result['image'].append(data['image_frames'][data['image_index'][row]])
            result['tasks'].append(data['task_frames'][data['task_index'][row]])
            result['task_mask'].append(data['task_mask_frames'][data['task_index'][row]])
        return {key: torch.as_tensor(np.stack(value).astype(np.float32), device=device)
                for key, value in result.items()}


def loss_components(model, batch, positive_weight, wrist_weight=4.):
    predicted = model(batch)
    # Joint errors are normalized policy actions; report radians separately.
    errors = (predicted['action'] - batch['action']).square()
    joint = errors.mean()
    wrist = errors[:, -2:].mean()
    joint_objective = errors[:, :-2].mean() + wrist_weight * wrist
    command = ((predicted['command'] - batch['command']) / model.command_limits).square().mean()
    fire = nn.functional.binary_cross_entropy_with_logits(
        predicted['fire_logit'], batch['fire'], pos_weight=positive_weight)
    residual = predicted['residual'][:, :-2].square().mean()
    candidate = predicted.get('wrist_candidate', predicted['action'][:, -2:])
    goal_error = (candidate - batch['wrist_goal']).square().mean(-1)
    goal = (goal_error * batch['wrist_goal_mask']).sum() / batch['wrist_goal_mask'].sum().clamp_min(1.)
    loss = joint_objective + .5 * command + .05 * fire + .1 * residual + goal
    return loss, {'loss': loss, 'joint_mse': joint, 'command_mse': command,
                  'fire_bce': fire, 'nonwrist_residual': residual, 'wrist_mse': wrist, 'wrist_goal_mse': goal,
                  'fire_predicted_rate': (predicted['fire_logit'] > 0).float().mean(),
                  'fire_label_rate': batch['fire'].mean()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--onnx', type=Path, default=Path('third_party/holosoma/fastsac_g1_29dof.onnx'))
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261126)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--unfreeze-gait', action='store_true')
    parser.add_argument('--positive-fraction', type=float, default=.125)
    parser.add_argument('--wrist-weight', type=float, default=4.)
    parser.add_argument('--wrist-error-limits', type=float, nargs=2)
    parser.add_argument('--wrist-residual-scale', type=float)
    parser.add_argument('--randomize-remaining', action='store_true', help='Prevent a fixed episode duration shortcut in the game features.')
    parser.add_argument('--randomize-elapsed', action='store_true', help='Randomize elapsed after startup; the teacher only uses initial 2.5s timing.')
    args = parser.parse_args()
    if not 0 <= args.positive_fraction <= 1 or args.wrist_weight <= 0:
        parser.error('positive-fraction must be in [0,1]; wrist-weight must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted({p for folder in args.data for p in folder.rglob('*.npz') if not p.name.endswith('.tmp.npz')})
    if len(files) < 2:
        raise ValueError('At least two independent episodes required for held-out validation')
    # Stable deterministic split recorded in manifest; do not random-split frames.
    split_rng = random.Random(args.seed)
    split_rng.shuffle(files)
    n_val = max(1, len(files) // 5)
    val_files, train_files = files[:n_val], files[n_val:]
    train, val = Episodes(train_files), Episodes(val_files)
    model = (FoundationNet.from_checkpoint(args.checkpoint) if args.checkpoint
             else FoundationNet(args.onnx))
    if args.wrist_error_limits:
        model.set_wrist_error_limits(args.wrist_error_limits)
    if args.wrist_residual_scale:
        model.residual_limits[-2:] = args.wrist_residual_scale
        model.model_config['wrist_residual_scale'] = args.wrist_residual_scale
    parity = validate_gait_import(args.onnx) if not args.checkpoint else None
    if parity and not parity['passed']:
        raise RuntimeError(f'ONNX transfer parity failed: {parity}')
    if args.unfreeze_gait:
        model.gait.requires_grad_(True)
        model.model_config['freeze_gait'] = False
    model.to(args.device)
    head = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('gait.')]
    groups = [{'params': head, 'lr': args.lr}]
    gait_params = [p for p in model.gait.parameters() if p.requires_grad]
    if gait_params:
        groups.append({'params': gait_params, 'lr': args.lr * .01})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    positives = sum(float(ep['fire'].sum()) for ep in train.episodes)
    positive_weight = torch.tensor(1. if args.positive_fraction > 0 else
        min(20., max(1., (train.ends[-1] - positives) / max(positives, 1.))), device=args.device)
    manifest = {'arguments': {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) else v
                              for k, v in vars(args).items()},
                'train_files': [str(p) for p in train_files], 'validation_files': [str(p) for p in val_files],
                'data_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'training_frames': int(train.ends[-1]), 'validation_frames': int(val.ends[-1]),
                'positive_fire_labels': positives, 'gait_parity': parity,
                'encoder_sha256': hashlib.sha256(Path(__file__).with_name('policy.py').read_bytes()).hexdigest(),
                'source_sha256': {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                  for file in sorted(Path(__file__).parent.glob('*.py'))},
                'torch': str(torch.__version__), 'cuda_device': torch.cuda.get_device_name(0) if 'cuda' in args.device else None,
                'stage': 'behavior_cloning', 'acceptance': 'unverified_requires_closed_loop'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rng = np.random.default_rng(args.seed)
    started, best = time.monotonic(), float('inf')
    save_checkpoint(args.output / 'initial.pt', model, metadata=manifest)
    with (args.output / 'metrics.jsonl').open('w', encoding='utf-8') as log:
        for step in range(1, args.steps + 1):
            model.train()
            batch = train.batch(args.batch_size, rng, args.device, args.positive_fraction)
            if args.randomize_remaining:
                batch['game'][:, 5] = torch.as_tensor(rng.uniform(0., 1., args.batch_size), device=args.device, dtype=torch.float32)
            if args.randomize_elapsed:
                elapsed = batch['game'][:, 6]
                randomized = torch.as_tensor(rng.uniform(2.5 / 60., 1., args.batch_size), device=args.device, dtype=torch.float32)
                batch['game'][:, 6] = torch.where(elapsed > 2.5 / 60., randomized, elapsed)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = loss_components(model, batch, positive_weight, args.wrist_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % args.save_every == 0 or step == args.steps:
                model.eval()
                with torch.inference_mode():
                    validation = [loss_components(model, val.batch(args.batch_size, rng, args.device), positive_weight, args.wrist_weight)[1]
                                  for _ in range(8)]
                record = {'step': step, 'wall_seconds': time.monotonic() - started,
                          'train': {key: float(value.detach()) for key, value in metrics.items()},
                          'validation': {key: float(torch.stack([v[key] for v in validation]).mean()) for key in metrics}}
                log.write(json.dumps(record) + '\n'); log.flush()
                print(json.dumps(record), flush=True)
                metadata = {**manifest, 'step': step, 'metrics': record}
                save_checkpoint(args.output / 'latest.pt', model, metadata=metadata)
                save_checkpoint(args.output / f'step_{step:06d}.pt', model, metadata=metadata)
                if record['validation']['loss'] < best:
                    best = record['validation']['loss']
                    save_checkpoint(args.output / 'best_offline.pt', model, metadata=metadata)
    (args.output / 'completed.json').write_text(json.dumps({'steps': args.steps, 'best_offline_loss': best,
        'wall_seconds': time.monotonic() - started, 'closed_loop_passed': False}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
