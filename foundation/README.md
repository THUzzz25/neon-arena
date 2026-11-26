# Foundation network experiments

This is an experimental training pipeline, not an accepted competition baseline.
The final actor consumes only public RGB-D, task reference images, proprioception,
game state and its own previous neural state. It directly produces all 29 joint
targets and a fire decision. Teacher detection, inverse kinematics and simulator
ray labels are used for training only. The full actor exports to a single ONNX
graph and runs without Torch or MuJoCo in the submission container.

The first curriculum is the legacy room with a passive standing opponent. Passing
that curriculum does not establish city navigation, mixed-v1 discrimination,
traffic avoidance or robustness against an attacking opponent. Task text is not
encoded in this first network; city task language and longer-term search memory
remain work for subsequent curricula.

## Server14

Workspace: `/share/public-local/caoxuanhe/neon-arena-foundation`.
Use `bash scripts/foundation_server14.sh ...` from that directory. The wrapper
selects physical GPU 0 and user-space OSMesa rendering, and limits CPU threads.
MuJoCo physics and software rendering run on CPU; network training runs on the
RTX 3090. The project virtual environment currently obtains Torch 2.10.0+cu126
through `shared_base.pth` from the existing flowmatching environment. Its own
MuJoCo/ONNX/ORT dependencies and rendering libraries are project-local.

## Reproducible workflow

```bash
# Stand, forward, left/right turn stability before training.
bash scripts/foundation_server14.sh -m foundation.rollout matrix \
  --modes stand forward left right --seeds 7 --duration 20 \
  --output artifacts/foundation/new_matrix

# Teacher data: current ray labels are training labels, never actor inputs.
bash scripts/foundation_server14.sh -m foundation.rollout collect \
  --modes vision --seeds 701 702 703 --duration 20 \
  --spawn-jitter .15 --yaw-jitter .12 --fire-labels ray-safe \
  --output artifacts/foundation/new_teacher_data

# Do not use fixed-command matrix episodes as conflicting task BC labels.
bash scripts/foundation_server14.sh -m foundation.train \
  --data artifacts/foundation/new_teacher_data \
  --output artifacts/foundation/new_training --steps 3000 \
  --wrist-error-limits .08 .05 --wrist-residual-scale 8 \
  --randomize-remaining --randomize-elapsed

# Real student actions in held-out physics; do not substitute offline loss.
bash scripts/foundation_server14.sh -m foundation.rollout evaluate \
  --checkpoint artifacts/foundation/new_training/latest.pt \
  --seeds 1001 1002 --spawn-swaps 0 1 --duration 60 \
  --spawn-jitter .3 --yaw-jitter .2 --output artifacts/foundation/new_evaluation
```

`foundation.iterate` automates a bounded sequence of corrective DAgger collection,
training, screening and longer acceptance. Its `--run-dir` must be new. Read
`status.json` for the current child process and stage; command logs and SHA256
manifests preserve provenance. `promoted.json` means the documented room gate
passed. `unmet.json`, a low loss, an exported diagnostic ZIP, or a completed
training process do not mean acceptance. Actual Torch and exported ONNX actors
must both pass the same held-out closed-loop gate. `--self-test` exercises the gate's
failure cases. The driver exports an accepted ZIP only after the room gate.

## Training details and known limitations

- The pretrained Holosoma gait is imported numerically into Torch; the first
  stages freeze its weights but keep gradients through its inputs. Optional
  `--unfreeze-gait` uses a smaller gait learning rate. This is one deployable
  neural graph with internal structure, not a deployed rule-based teacher.
- Encoder recurrence retains the raw neural gait output, separately from wrist
  corrections. Reset clears all history. Camera frames remain at 10 Hz while
  control and recurrent updates run at 50 Hz.
- Wrist error limits are relative to measured position, not a guarantee about
  target slew or stability. The pre-limit wrist candidate has an auxiliary IK
  target loss to preserve gradients when limiting is active.
- `fire-labels ray-safe` labels the current physical muzzle ray, including task
  ownership and cooldown. Probe traces are removed and actual executed actions
  remain unchanged. The deployment actor does not access the ray caster.
- Dataset images/references are deduplicated; labels remain 50 Hz. Recordings
  terminate at a fall, death or reset. Episode-level offline splits do not
  replace independent closed-loop seeds, especially after checkpoint reuse.
- Historical `dagger_v1` used a counterfactual wrist smoothing history. It is
  excluded from the current automatic pipeline. Current DAgger aligns the
  teacher with the actually executed wrist and raw gait histories.
- Training returns are raw score changes. This implementation is supervised
  BC/DAgger; PPO and reward shaping have not been implemented or validated.

Run the three `tests/test_foundation_*.py` modules in the prepared environment.
They exercise gait parity, recurrence, public input order, action labels,
privileged-label isolation, randomized resets, and an ONNX submission in a
subprocess that forbids Torch/MuJoCo imports. Export includes the original
Holosoma license and source/model hashes.
