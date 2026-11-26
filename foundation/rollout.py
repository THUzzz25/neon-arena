"""50 Hz teacher collection and closed-loop foundation policy evaluation.

Examples (run from the repository root)::

    python -m foundation.rollout matrix --seeds 7 --duration 15
    python -m foundation.rollout collect --seeds 101 102 --modes stand vision
    python -m foundation.rollout evaluate --checkpoint model.pt --seeds 1001 1002
    python -m foundation.rollout evaluate --onnx-model model.onnx --onnx-metadata metadata.json

Collection uses only public observations as model inputs. Private simulator state
is used exclusively for evaluation metrics. Images and task references are stored
once per change, with per-control-step indices; all action labels remain 50 Hz.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from embodied_arena.config import MatchConfig
from embodied_arena.engine import Arena
from embodied_arena.locomotion import HolosomaWalking
from embodied_arena.policies import DiagnosticVision, WalkInPlace


COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.18, 0.0, 0.0),
    "left": (0.0, 0.0, 0.15),
    "right": (0.0, 0.0, -0.15),
}


class CommandTeacher:
    """Unmodified pretrained gait driven by a fixed, conservative command."""

    def __init__(self, command):
        self.walk = HolosomaWalking()
        self.command = np.asarray(command, dtype=np.float32)
        self.reset_count = -1

    def act(self, observation):
        if observation["reset_count"] != self.reset_count:
            self.walk.reset()
            self.reset_count = observation["reset_count"]
        p = observation["proprio"]
        target = self.walk.act(p["q"], p["dq"], p["gyro"], p["base_quat"],
                               observation["time"], self.command)
        return {"joint_targets": target, "fire": False}


def make_teacher(mode: str, config: MatchConfig):
    if mode == "vision":
        public_config = asdict(config)
        public_config.pop("seed", None)
        return DiagnosticVision(player_id=0, config=public_config)
    return CommandTeacher(COMMANDS[mode])


def consumed_command(policy, observation) -> np.ndarray:
    """Snapshot before act(): DiagnosticVision changes its NEXT command in act."""
    if isinstance(policy, DiagnosticVision) and observation["reset_count"] != policy.reset_count:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(getattr(policy, "command", [0., 0., 0.]), dtype=np.float32).copy()


class EpisodeBuffer:
    def __init__(self):
        self.rows: dict[str, list] = {key: [] for key in
            ("proprio", "game", "action", "command", "fire", "wrist_goal", "reward", "time", "failure",
             "gait_action", "teacher_action", "executed_action", "executed_gait_action",
             "executed_command", "executed_fire", "teacher_fire_request", "teacher_used", "image_index", "task_index")}
        self.images: list[np.ndarray] = []
        self.tasks: list[np.ndarray] = []
        self.task_masks: list[np.ndarray] = []
        self.image_lookup: dict[tuple, int] = {}
        self.task_lookup: dict[tuple, int] = {}

    def append(self, encoded, observation, action, command, fire, reward, failure, gait_action,
               *, executed_action=None, executed_gait_action=None, executed_command=None,
               executed_fire=None, teacher_fire_request=None, teacher_used=True, wrist_goal=None):
        visual = observation.get("visual")
        image_key = (observation["reset_count"], None if visual is None else visual["time"])
        task_key = tuple((task["id"], task["kind"]) for task in observation["tasks"])
        if image_key not in self.image_lookup:
            self.image_lookup[image_key] = len(self.images)
            self.images.append(np.asarray(encoded["image"], dtype=np.float16).copy())
        if task_key not in self.task_lookup:
            self.task_lookup[task_key] = len(self.tasks)
            self.tasks.append(np.asarray(encoded["tasks"], dtype=np.float16).copy())
            self.task_masks.append(np.asarray(encoded["task_mask"], dtype=np.float32).copy())
        values = dict(proprio=encoded["proprio"], game=encoded["game"], action=action,
                      command=command, fire=float(fire),
                      wrist_goal=np.asarray(action)[-2:] if wrist_goal is None else wrist_goal,
                      reward=reward, time=observation["time"],
                      failure=bool(failure), gait_action=gait_action,
                      teacher_action=action,
                      executed_action=action if executed_action is None else executed_action,
                      executed_gait_action=gait_action if executed_gait_action is None else executed_gait_action,
                      executed_command=command if executed_command is None else executed_command,
                      executed_fire=float(fire if executed_fire is None else executed_fire),
                      teacher_fire_request=float(fire if teacher_fire_request is None else teacher_fire_request),
                      teacher_used=float(teacher_used),
                      image_index=self.image_lookup[image_key], task_index=self.task_lookup[task_key])
        for key, value in values.items():
            self.rows[key].append(np.array(value, copy=True))

    def save(self, path: str | Path, metadata: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(values, dtype=(np.int32 if key.endswith("_index") else
                  np.bool_ if key == "failure" else np.float32)) for key, values in self.rows.items()}
        arrays.update(image_frames=np.asarray(self.images, dtype=np.float16),
                      task_frames=np.asarray(self.tasks, dtype=np.float16),
                      task_mask_frames=np.asarray(self.task_masks, dtype=np.float32),
                      metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)


def load_episode(path: str | Path, *, expand: bool = False) -> dict:
    """Load compact arrays; expand=True also materializes every model input."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        result["metadata"] = json.loads(str(archive["metadata"].item()))
    if expand:
        result["image"] = result["image_frames"][result["image_index"]].astype(np.float32)
        result["tasks"] = result["task_frames"][result["task_index"]].astype(np.float32)
        result["task_mask"] = result["task_mask_frames"][result["task_index"]]
    return result


def apply_spawn_jitter(arena, spawn_jitter: float = 0., yaw_jitter: float = 0.) -> dict:
    """Seeded, collision-checked room perturbations; keep base z and velocities."""
    if not np.isfinite([spawn_jitter, yaw_jitter]).all() or min(spawn_jitter, yaw_jitter) < 0:
        raise ValueError("Spawn/yaw jitter must be finite and nonnegative")
    result = {"xy_bound_m": float(spawn_jitter), "yaw_bound_rad": float(yaw_jitter), "offsets": []}
    if not spawn_jitter and not yaw_jitter:
        return result
    if arena.config.scene != "room":
        raise ValueError("Spawn jitter is supported only in room scenes")
    if arena.time != 0. or getattr(arena, "_foundation_spawn_jitter_applied", False):
        raise ValueError("Spawn jitter must be applied exactly once, before stepping")
    import mujoco
    rng = np.random.default_rng(np.random.SeedSequence([arena.config.seed, int(arena.config.spawn_swap), 314159]))
    original = [arena.data.qpos[r.root_q:r.root_q + 7].copy() for r in arena.robots]
    for attempt in range(100):
        offsets = []
        inside = True
        for robot, pose in zip(arena.robots, original):
            dx, dy = rng.uniform(-spawn_jitter, spawn_jitter, size=2)
            yaw = float(rng.uniform(-yaw_jitter, yaw_jitter))
            sampled = pose.copy()
            sampled[:2] += [dx, dy]
            mujoco.mju_mulQuat(sampled[3:7], np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]), pose[3:7])
            arena.data.qpos[robot.root_q:robot.root_q + 7] = sampled
            inside &= abs(sampled[0]) < 3.35 and abs(sampled[1]) < 2.35
            offsets.append({"player": robot.pid, "dx_m": float(dx), "dy_m": float(dy),
                            "yaw_rad": yaw, "initial_pose": sampled.tolist()})
        mujoco.mj_forward(arena.model, arena.data)
        blocked = False
        for contact in arena.data.contact:
            names = [arena.model.geom(int(gid)).name or "" for gid in (contact.geom1, contact.geom2)]
            owners = [next((r.pid for r in arena.robots if name.startswith(r.prefix)), None) for name in names]
            if contact.dist < -1e-4 and "floor" not in names and any(owner is not None for owner in owners) and owners[0] != owners[1]:
                blocked = True
                break
        if inside and not blocked:
            for robot in arena.robots:
                robot.spawn = arena.data.qpos[robot.root_q:robot.root_q + 7].copy()
            arena.images.clear()
            arena.traces.clear()
            arena._foundation_spawn_jitter_applied = True
            result.update(offsets=offsets, attempts=attempt + 1)
            return result
    for robot, pose in zip(arena.robots, original):
        arena.data.qpos[robot.root_q:robot.root_q + 7] = pose
    mujoco.mj_forward(arena.model, arena.data)
    raise ValueError("Could not sample a collision-free room spawn in 100 attempts")


def dagger_teacher_act(policy, observation, previous_executed_target):
    """Base wrist smoothing on executed targets, including the teacher's reset tick."""
    if not all(hasattr(policy, name) for name in ("arm", "target_command", "robot")):
        return policy.act(observation)
    names = observation["proprio"]["joint_names"]
    indices = [names.index(policy.robot.names[int(i)]) for i in policy.arm]
    source = observation["proprio"]["q"] if previous_executed_target is None else previous_executed_target
    origin = np.asarray(source)[indices].copy()
    original_walk = policy.walk.act

    def walk_with_origin(*args, **kwargs):
        # DiagnosticVision resets target_command before calling walk.act. Setting
        # here preserves that reset while honoring measured first-step wrists.
        policy.target_command = origin.copy()
        return original_walk(*args, **kwargs)

    policy.walk.act = walk_with_origin
    try:
        return policy.act(observation)
    finally:
        policy.walk.act = original_walk


def ray_safe_fire_label(arena, pid: int = 0) -> bool:
    """Privileged training label at CURRENT muzzle pose; no shot is executed.

    cast_shot adds a visualization trace but performs no rules/physics update.
    Remove exactly the probe's appended traces, preserving any existing traces.
    """
    if not arena.referee.can_fire(pid, arena.time):
        return False
    trace_count = len(arena.shot_traces)
    try:
        hit = arena.cast_shot(pid)
    finally:
        del arena.shot_traces[trace_count:]
    if hit.kind != "target":
        return False
    target = arena.referee.targets.get(str(hit.id))
    return bool(target is not None and target.active and target.owner in (None, pid))


def wrist_goal_label(policy, encoder, fallback_action) -> np.ndarray:
    """Unsmoothed own-robot IK goal, in canonical normalized wrist coordinates."""
    if isinstance(policy, DiagnosticVision) and policy.aim_point is not None:
        # target contains the IK result before target_command rate limiting.
        # It was derived solely from public RGB-D and own robot kinematics.
        desired = encoder.targets_to_action(policy.target, policy.robot.names)
        return np.asarray(desired[-2:], dtype=np.float32).copy()
    return np.asarray(fallback_action[-2:], dtype=np.float32).copy()


def run_episode(config: MatchConfig, policy, *, mode: str = "custom", collect_path=None,
                cameras: bool = True, stop_on_failure: bool = True,
                opponent=None, arena_factory: Callable = Arena, encoder_factory=None,
                command_reader: Callable = consumed_command, rollout_policy=None,
                teacher_probability: float = 0., spawn_jitter: float = 0., yaw_jitter: float = 0.,
                provenance: dict | None = None, fire_labels: str = "teacher") -> dict:
    """Run actual closed-loop physics, never count post-reset training success.

    A caller may supply any teacher policy and a command_reader that captures the
    command consumed by that teacher. Student evaluation does not need an encoder.
    """
    if not 0. <= teacher_probability <= 1.:
        raise ValueError("teacher_probability must be between zero and one")
    if rollout_policy is not None and collect_path is None:
        raise ValueError("DAgger rollout_policy requires collect_path")
    if fire_labels not in ("teacher", "ray-safe"):
        raise ValueError("fire_labels must be teacher or ray-safe")
    if fire_labels == "ray-safe" and collect_path is None:
        raise ValueError("Ray-safe firing labels are training-only and require collect_path")
    if collect_path is not None:
        if encoder_factory is None:
            from .policy import ObservationEncoder
            encoder_factory = ObservationEncoder
        encoder = encoder_factory()
        buffer = EpisodeBuffer()
    else:
        encoder = buffer = None
    opponent_nonvisual = opponent is None
    opponent = opponent if opponent is not None else WalkInPlace(player_id=1)
    arena = arena_factory(config, cameras=cameras)
    # Engine.record normally copies every qpos/qvel/control state. Training does
    # not save replay trajectories; do not accumulate hundreds of MB per worker.
    arena.record = lambda: None
    metrics = {"mode": mode, "config": asdict(config), "control_steps": 0,
               "score": 0.0, "health_lost": 0.0, "falls": 0, "misses": 0,
               "traffic_collisions": 0, "deaths": 0, "resets": 0,
               "invalid_actions": 0, "target_completions": 0, "shots": 0,
               "min_base_height": float("inf"), "min_up_z": 1.0,
               "termination": "duration", "events": [], "cameras": cameras,
               "observation_source": "public_only", "control_hz": 1 / config.control_dt,
               "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
               "provenance": dict(provenance or {}),
               "fire_label_source": fire_labels,
               "wrist_goal_source": "unsmoothed teacher IK from public RGB-D and own kinematics; fallback to action wrists",
               "privileged_training_labels": (["current muzzle ray hit valid active own/public target and can_fire"]
                                              if fire_labels == "ray-safe" else []),
               "teacher_history": "raw gait previous action; labels use executed clipped targets",
               "dagger": rollout_policy is not None, "teacher_probability": teacher_probability if rollout_policy is not None else 1.,
               "opponent_nonvisual": opponent_nonvisual,
               "room_randomization": "seed jitters cup lateral positions by +/-0.08m; optional spawn/yaw offsets recorded separately"}
    inference_times = []
    residual_max = np.zeros(29)
    mix_rng = np.random.default_rng(config.seed * 2 + int(config.spawn_swap))
    started = time.perf_counter()
    try:
        metrics["spawn_randomization"] = apply_spawn_jitter(arena, spawn_jitter, yaw_jitter)
        previous_executed_target = None
        done = False
        while not done:
            observation = arena.observation(0)
            encoded = encoder.encode(observation) if encoder is not None else None
            if rollout_policy is not None:
                # Query the expert from the recurrent state that actually reached
                # this observation, rather than its counterfactual last action.
                for attr in ("last", "phase", "standing"):
                    value = getattr(encoder, attr)
                    setattr(policy.walk, attr, value.copy() if isinstance(value, np.ndarray) else value)
            command = command_reader(policy, observation)
            call_started = time.perf_counter()
            teacher_action = (dagger_teacher_act(policy, observation, previous_executed_target)
                              if rollout_policy is not None else policy.act(observation))
            teacher_raw_gait = getattr(getattr(policy, "walk", None), "last", None)
            if teacher_raw_gait is not None:
                teacher_raw_gait = np.asarray(teacher_raw_gait, dtype=np.float32).copy()
            action = teacher_action
            actual_command = command
            actual_gait = teacher_raw_gait
            teacher_used = True
            if rollout_policy is not None:
                # Sync before act; the student's own adapter advances its state
                # after inference, so the collector overrides it on the next tick.
                rollout_policy.encoder.encode(observation)
                for attr in ("last", "phase", "standing"):
                    value = getattr(encoder, attr)
                    setattr(rollout_policy.encoder, attr, value.copy() if isinstance(value, np.ndarray) else value)
                student_action = rollout_policy.act(observation)
                teacher_used = bool(mix_rng.random() < teacher_probability)
                if not teacher_used:
                    action = student_action
                    actual_command = np.asarray(rollout_policy.last_prediction["command"], dtype=np.float32).copy()
                    actual_gait = np.asarray(rollout_policy.last_prediction["gait_action"], dtype=np.float32).copy()
            inference_times.append(time.perf_counter() - call_started)
            # Capture before step: a physical reset replaces last_targets with
            # home, which must not overwrite the action that caused the failure.
            executed_target = np.clip(np.asarray(action["joint_targets"]),
                                      arena.robots[0].limits[:, 0], arena.robots[0].limits[:, 1]).copy()
            if opponent_nonvisual:
                previous_cameras = arena.cameras
                try:
                    arena.cameras = False
                    opponent_observation = arena.observation(1)
                finally:
                    arena.cameras = previous_cameras
            else:
                opponent_observation = arena.observation(1)
            other_action = opponent.act(opponent_observation)
            previous_executed_target = executed_target.copy()
            previous_health = observation["game"]["health"]
            previous_resets = observation["reset_count"]
            fire_label = (ray_safe_fire_label(arena) if fire_labels == "ray-safe"
                          else bool(teacher_action.get("fire", False)))
            rewards, done = arena.step([action, other_action])
            new_observation = arena.observation(0)
            events = [dict(event) for event in arena.referee.events if
                      event.get("player") == 0 or event.get("victim") == 0]
            own_events = [event for event in events if event.get("player") == 0]
            kinds = [event["type"] for event in own_events]
            reset_delta = new_observation["reset_count"] - previous_resets
            failed = "fall" in kinds or "death" in kinds or reset_delta > 0
            metrics["control_steps"] += 1
            metrics["score"] += float(rewards[0])
            # Full-health restoration can never yield a positive reward/negative loss.
            metrics["health_lost"] += max(0., previous_health - new_observation["game"]["health"])
            metrics["resets"] += reset_delta
            for key, event_type in (("falls", "fall"), ("misses", "miss"),
                                    ("traffic_collisions", "traffic_collision"), ("deaths", "death"),
                                    ("invalid_actions", "invalid_action"), ("target_completions", "target_completed")):
                metrics[key] += kinds.count(event_type)
            metrics["events"].extend(events)
            metrics["shots"] += sum(trace["player"] == 0 for trace in arena.shot_traces)
            robot = arena.robots[0]
            metrics["min_base_height"] = min(metrics["min_base_height"], float(arena.data.qpos[robot.root_q + 2]))
            metrics["min_up_z"] = min(metrics["min_up_z"], float(arena.data.xmat[robot.body].reshape(3, 3)[2, 2]))
            if buffer is not None:
                # Arena clips joint positions before execution. Use that accepted
                # value rather than an out-of-range policy request as the label.
                executed = encoder.targets_to_action(executed_target, observation["proprio"]["joint_names"])
                label = encoder.targets_to_action(teacher_action["joint_targets"], observation["proprio"]["joint_names"])
                gait = label.copy() if teacher_raw_gait is None else teacher_raw_gait
                actual_gait = executed.copy() if actual_gait is None else actual_gait
                residual_max = np.maximum(residual_max, np.abs(label - gait))
                buffer.append(encoded, observation, label, command, fire_label, rewards[0], failed, gait,
                              executed_action=executed, executed_gait_action=actual_gait,
                              executed_command=actual_command, executed_fire=action.get("fire", False),
                              teacher_fire_request=teacher_action.get("fire", False), teacher_used=teacher_used,
                              wrist_goal=wrist_goal_label(policy, encoder, label))
                # Match Holosoma's recurrence: wrist IK/residual and position
                # clipping are not fed back into its previous-action channel.
                encoder.commit(actual_gait, actual_command)
            arena.referee.events.clear()
            arena.shot_traces.clear()
            if failed and stop_on_failure:
                metrics["termination"] = "fall" if "fall" in kinds else "death" if "death" in kinds else "reset"
                break
        metrics.update(sim_seconds=float(arena.time), wall_seconds=time.perf_counter() - started,
                       final_health=float(arena.referee.players[0].health),
                       full_episode=bool(done),
                       no_fall=metrics["falls"] == 0,
                       policy_ms_mean=float(np.mean(inference_times) * 1000),
                       policy_ms_p95=float(np.percentile(inference_times, 95) * 1000),
                       policy_ms_max=float(np.max(inference_times) * 1000),
                       teacher_executed_minus_raw_gait_absmax=residual_max.tolist())
        if buffer is not None:
            metrics["action_joint_names"] = list(getattr(encoder, "joint_names", observation["proprio"]["joint_names"]))
            metrics["action_scale"] = float(getattr(encoder, "action_scale", .25))
            buffer.save(collect_path, metrics)
            metrics["dataset"] = str(collect_path)
        return metrics
    finally:
        arena.close()


def aggregate(episodes: list[dict]) -> dict:
    count = len(episodes)
    return {"episodes": count,
            "full_episode_rate": sum(e["full_episode"] for e in episodes) / max(1, count),
            "no_fall_rate": sum(e["no_fall"] for e in episodes) / max(1, count),
            **{key: sum(e[key] for e in episodes) for key in
               ("score", "falls", "misses", "traffic_collisions", "deaths", "resets", "invalid_actions", "health_lost")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("operation", choices=("matrix", "collect", "evaluate"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/foundation/rollouts"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--modes", nargs="+", choices=(*COMMANDS, "vision"), default=["stand", "forward", "left", "right", "vision"])
    parser.add_argument("--duration", type=float, default=15.)
    parser.add_argument("--scene", choices=("room", "cyber_city"), default="room")
    parser.add_argument("--task-set", choices=("legacy", "mixed-v1"), default="legacy")
    parser.add_argument("--spawn-swaps", type=int, choices=(0, 1), nargs="+", default=[0])
    parser.add_argument("--traffic", action="store_true")
    parser.add_argument("--spawn-jitter", type=float, default=0., help="Room only: maximum absolute initial x/y offset, in metres.")
    parser.add_argument("--yaw-jitter", type=float, default=0., help="Room only: maximum absolute initial yaw offset, in radians.")
    evaluation_model = parser.add_mutually_exclusive_group()
    evaluation_model.add_argument("--checkpoint", type=Path)
    evaluation_model.add_argument("--onnx-model", type=Path, help="Evaluate the actual exported graph using ONNX Runtime CPU.")
    parser.add_argument("--onnx-metadata", type=Path, help="ONNX metadata JSON; default is metadata.json beside the model.")
    parser.add_argument("--student-checkpoint", type=Path, help="Collect DAgger labels on this student's closed-loop states.")
    parser.add_argument("--teacher-probability", type=float, default=0., help="DAgger: probability of executing teacher action, using a seed-deterministic mix.")
    parser.add_argument("--fire-labels", choices=("teacher", "ray-safe"), default="teacher",
                        help="Collection only: ray-safe uses privileged current-pose validity for firing labels, without changing executed actions.")
    parser.add_argument("--device", default="cpu", help="Torch device; ONNX evaluation supports cpu only.")
    parser.add_argument("--continue-after-failure", action="store_true", help="Evaluation only: continue after falls/death, with honest failure counts.")
    args = parser.parse_args(argv)
    if args.operation == "evaluate" and args.checkpoint is None and args.onnx_model is None:
        parser.error("evaluate requires --checkpoint or --onnx-model")
    if args.onnx_model is not None and args.operation != "evaluate":
        parser.error("--onnx-model is only valid for evaluate")
    if args.onnx_metadata is not None and args.onnx_model is None:
        parser.error("--onnx-metadata requires --onnx-model")
    if args.onnx_model is not None and args.device != "cpu":
        parser.error("ONNX evaluation uses CPUExecutionProvider; --device must be cpu")
    if args.operation == "collect" and args.continue_after_failure:
        parser.error("collection always stops at fall/death/reset")
    if args.student_checkpoint is not None and args.operation != "collect":
        parser.error("--student-checkpoint is only valid for collect")
    if args.fire_labels != "teacher" and args.operation != "collect":
        parser.error("--fire-labels ray-safe is only valid for collect")
    if not 0. <= args.teacher_probability <= 1.:
        parser.error("--teacher-probability must be between zero and one")
    if min(args.spawn_jitter, args.yaw_jitter) < 0. or not np.isfinite([args.spawn_jitter, args.yaw_jitter]).all():
        parser.error("jitter values must be finite and nonnegative")
    if args.scene != "room" and (args.spawn_jitter or args.yaw_jitter):
        parser.error("spawn/yaw jitter is supported only for --scene room")
    onnx_provenance = {}
    if args.onnx_model is not None:
        args.onnx_metadata = args.onnx_metadata or args.onnx_model.with_name("metadata.json")
        onnx_provenance = {
            "inference_backend": "onnxruntime", "inference_device": "cpu",
            "onnx_model": str(args.onnx_model.resolve()),
            "onnx_metadata": str(args.onnx_metadata.resolve()),
            "onnx_sha256": hashlib.sha256(args.onnx_model.read_bytes()).hexdigest(),
            "onnx_metadata_sha256": hashlib.sha256(args.onnx_metadata.read_bytes()).hexdigest(),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    modes = ["student"] if args.operation == "evaluate" else args.modes
    for seed in args.seeds:
        for swap in args.spawn_swaps:
            config = MatchConfig(scene=args.scene, task_set=args.task_set, seed=seed,
                                 spawn_swap=bool(swap), duration=args.duration,
                                 traffic_enabled=args.traffic, render_shadows=False)
            for mode in modes:
                name = f"{args.scene}-{args.task_set}-{mode}-seed{seed}-swap{swap}"
                if args.operation == "evaluate":
                    if args.onnx_model is not None:
                        from .onnx_policy import Policy as OnnxPolicy
                        policy = OnnxPolicy(player_id=0, config=asdict(config),
                                            model_path=args.onnx_model, metadata_path=args.onnx_metadata)
                    else:
                        from .policy import Policy
                        policy = Policy(player_id=0, config=asdict(config), checkpoint=args.checkpoint, device=args.device)
                else:
                    policy = make_teacher(mode, config)
                rollout_policy = None
                if args.student_checkpoint is not None:
                    from .policy import Policy
                    rollout_policy = Policy(player_id=0, checkpoint=args.student_checkpoint, device=args.device)
                summary = run_episode(config, policy, mode=mode,
                    collect_path=args.output / f"{name}.npz" if args.operation == "collect" else None,
                    cameras=args.operation != "matrix" or mode == "vision",
                    stop_on_failure=not args.continue_after_failure,
                    rollout_policy=rollout_policy, teacher_probability=args.teacher_probability,
                    spawn_jitter=args.spawn_jitter, yaw_jitter=args.yaw_jitter,
                    fire_labels=args.fire_labels,
                    provenance={"checkpoint": str(args.checkpoint) if args.checkpoint else None,
                                "student_checkpoint": str(args.student_checkpoint) if args.student_checkpoint else None,
                                **onnx_provenance})
                summary.update(onnx_provenance)
                if args.student_checkpoint is not None:
                    summary["student_checkpoint"] = str(args.student_checkpoint)
                if args.checkpoint is not None:
                    summary["checkpoint"] = str(args.checkpoint)
                summaries.append(summary)
                (args.output / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                (args.output / "summary.json").write_text(json.dumps({"aggregate": aggregate(summaries), "episodes": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
                print(json.dumps({"episode": name, **{key: summary[key] for key in
                    ("score", "falls", "misses", "resets", "sim_seconds", "wall_seconds", "termination")}}), flush=True)


if __name__ == "__main__":
    main()
