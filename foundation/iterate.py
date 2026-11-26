"""Bounded, auditable DAgger/BC iterations with real closed-loop promotion gates.

Run in the prepared server environment (its MuJoCo renderer variables are kept)::

    python -m foundation.iterate --run-dir artifacts/foundation/iteration-v1 \
      --initial-checkpoint artifacts/foundation/bc_goal/latest.pt \
      --teacher-data artifacts/foundation/teacher_goal_v4

Exit 0 means both Torch and exported ONNX passed the legacy-room gate.
Exit 2 means the allotted rounds finished without meeting that gate. Neither is
a claim of city navigation or adversarial-opponent acceptance.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCOPE = "legacy room; passive standing opponent; randomized spawn; no city acceptance"
PROBABILITIES = (.25, .1, 0.)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assess_promotion(report, *, expected_seeds, duration, swaps=(0, 1),
                     success_rate=.8, max_mean_health_loss=5., expected_policy_class=None):
    """Fail closed: a missing/partial report, zero score, or any fall cannot pass."""
    episodes = report.get("episodes", []) if isinstance(report, dict) else []
    expected = {(int(seed), bool(swap)) for seed in expected_seeds for swap in swaps}
    reasons, observed = [], []
    positive, health_loss = 0, 0.
    totals = {name: 0. for name in ("falls", "deaths", "invalid_actions", "resets", "score")}
    if not isinstance(episodes, list) or len(episodes) != len(expected) or not expected:
        reasons.append("Wrong episode count or empty evaluation")
        episodes = episodes if isinstance(episodes, list) else []
    for index, episode in enumerate(episodes):
        try:
            config = episode["config"]
            observed.append((int(config["seed"]), bool(config["spawn_swap"])))
            if config["scene"] != "room" or config["task_set"] != "legacy":
                reasons.append(f"Episode {index}: wrong evaluation scene/task set")
            configured_duration = float(config["duration"])
            simulated_duration = float(episode["sim_seconds"])
            if not math.isfinite(configured_duration) or not math.isfinite(simulated_duration):
                raise ValueError("Nonfinite duration")
            if abs(configured_duration - duration) > 1e-6:
                reasons.append(f"Episode {index}: wrong configured duration")
            if (episode["mode"] != "student" or episode["observation_source"] != "public_only"
                    or not episode["cameras"] or not episode.get("opponent_nonvisual", False)):
                reasons.append(f"Episode {index}: wrong deployed-policy evaluation conditions")
            if expected_policy_class is not None and episode.get("policy_class") != expected_policy_class:
                reasons.append(f"Episode {index}: policy backend is not {expected_policy_class}")
            if not episode["full_episode"] or abs(simulated_duration - duration) > .021:
                reasons.append(f"Episode {index}: did not complete the required duration")
            for name in totals:
                value = float(episode[name])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {name}")
                totals[name] += value
            damage = float(episode["health_lost"])
            if not math.isfinite(damage) or damage < 0:
                raise ValueError("Invalid health loss")
            health_loss += damage
            positive += float(episode["score"]) > 0.
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            reasons.append(f"Episode {index}: incomplete or invalid evidence ({error})")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        reasons.append("Missing, duplicate, or unexpected seed/spawn pairs")
    for name in ("falls", "deaths", "invalid_actions", "resets"):
        if totals[name] != 0:
            reasons.append(f"Nonzero {name}: {totals[name]:g}")
    rate = positive / max(1, len(episodes))
    mean_damage = health_loss / max(1, len(episodes))
    if rate + 1e-12 < success_rate:
        reasons.append(f"Positive-score episode rate {rate:.3f} < {success_rate:.3f}")
    if mean_damage > max_mean_health_loss:
        reasons.append(f"Mean health loss {mean_damage:.3f} > {max_mean_health_loss:.3f}")
    return {"passed": not reasons, "scope": SCOPE, "reasons": reasons,
            "expected_policy_class": expected_policy_class,
            "expected_episodes": len(expected), "observed_episodes": len(episodes),
            "positive_score_rate": rate, "mean_health_loss": mean_damage, "totals": totals,
            "requirements": {"positive_score_rate_at_least": success_rate,
                             "mean_health_loss_at_most": max_mean_health_loss,
                             "zero": ["falls", "deaths", "invalid_actions", "resets"],
                             "all_episodes_complete": True, "duration_seconds": duration}}


def both_backends_pass(torch_gate, onnx_gate):
    """A numerical export check or one passing inference backend is insufficient."""
    return (isinstance(torch_gate, dict) and isinstance(onnx_gate, dict)
            and torch_gate.get("passed") is True and onnx_gate.get("passed") is True
            and torch_gate.get("expected_policy_class") == "foundation.policy.Policy"
            and onnx_gate.get("expected_policy_class") == "foundation.onnx_policy.Policy")


def audit_data(directories):
    """Accept only explicit, current ray-safe/wrist-goal data; never scan a parent indiscriminately."""
    import numpy as np
    files, seeds = {}, set()
    required = {"proprio", "game", "image_frames", "image_index", "task_frames", "task_mask_frames",
                "task_index", "action", "command", "fire", "wrist_goal", "metadata"}
    for directory in directories:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ValueError(f"Training data directory does not exist: {directory}")
        paths = sorted(directory.rglob("*.npz"))
        if not paths:
            raise ValueError(f"No episodes in data directory: {directory}")
        for path in paths:
            if "dagger_v1" in path.parts:
                raise ValueError(f"Historical dagger_v1 is explicitly excluded: {path}")
            with np.load(path, allow_pickle=False) as archive:
                if not required.issubset(archive.files):
                    raise ValueError(f"Incomplete/new-schema data required (including wrist_goal): {path}")
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("mode") != "vision" or metadata.get("fire_label_source") != "ray-safe":
                    raise ValueError(f"Only vision teacher with ray-safe labels is allowed: {path}")
                if metadata.get("observation_source") != "public_only":
                    raise ValueError(f"Public-only actor input provenance missing: {path}")
                if archive["game"].shape[1:] != (8,) or archive["wrist_goal"].shape[1:] != (2,):
                    raise ValueError(f"Invalid feature/label shapes: {path}")
                if len(archive["action"]) < 1:
                    raise ValueError(f"Empty episode: {path}")
                seeds.add(int(metadata["config"]["seed"]))
            files[str(path)] = digest(path)
    return {"files": files, "seeds": sorted(seeds), "episode_count": len(files)}


class Driver:
    def __init__(self, args, data_audit):
        self.args = args
        self.run_dir = args.run_dir.resolve()
        self.started = time.monotonic()
        self.deadline = self.started + args.max_wall_seconds
        self.env = os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES="0", MUJOCO_EGL_DEVICE_ID="0",
                        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        self.active = None
        self.state = {"status": "starting", "stage": "preflight", "round": 0,
                      "driver_pid": os.getpid(), "child_pid": None, "started_at": now(),
                      "scope": SCOPE, "gpu_visible": "0", "accepted": False}
        atomic_json(self.run_dir / "manifest.json", {
            "arguments": {key: [str(item) for item in value] if isinstance(value, list) else
                          str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "initial_checkpoint_sha256": digest(args.initial_checkpoint), "input_data": data_audit,
            "source_sha256": {path.name: digest(path) for path in sorted((ROOT / "foundation").glob("*.py"))},
            "runtime": {key: self.env.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID")},
            "scope": SCOPE, "promotion_backends": ["torch", "onnxruntime_cpu"],
            "selection": "Each final training-step checkpoint and export stays a candidate until both inference backends pass closed-loop gates"})
        self.update()

    def update(self, **fields):
        self.state.update(fields, updated_at=now(), elapsed_seconds=time.monotonic() - self.started)
        atomic_json(self.run_dir / "status.json", self.state)

    def event(self, **fields):
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), **fields}, ensure_ascii=False) + "\n")

    def terminate_owned_child(self):
        child = self.active
        if child is None or child.poll() is not None:
            return
        # Every POSIX child starts its own group; only that known group is stopped.
        if os.name == "posix":
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                child.kill()
            child.wait(timeout=10)

    def command(self, stage, arguments):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Total iteration time budget reached")
        command = [sys.executable, *map(str, arguments)]
        logfile = self.run_dir / (stage + ".log")
        stage_started = time.monotonic()
        stage_deadline = min(self.deadline, stage_started + self.args.stage_timeout)
        self.event(event="child_start", stage=stage, command=command, log=str(logfile))
        with logfile.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"started_at": now(), "command": command}) + "\n")
            log.flush()
            self.active = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            self.update(status="running", stage=stage, child_pid=self.active.pid, child_command=command, log=str(logfile))
            try:
                while self.active.poll() is None:
                    remaining = stage_deadline - time.monotonic()
                    if remaining <= 0:
                        self.terminate_owned_child()
                        raise TimeoutError(f"Time limit reached in {stage}")
                    try:
                        self.active.wait(timeout=min(15., remaining))
                    except subprocess.TimeoutExpired:
                        self.update()
                code = self.active.returncode
            except BaseException:
                self.terminate_owned_child()
                raise
            finally:
                self.active = None
                self.update(child_pid=None)
        self.event(event="child_end", stage=stage, exit_code=code, wall_seconds=time.monotonic() - stage_started)
        if code:
            raise RuntimeError(f"Child stage {stage} failed with exit code {code}; see {logfile}")

    def run(self):
        args = self.args
        self.command("gpu_probe", ["-c", "import json,torch; assert torch.cuda.is_available(); "
                     "assert torch.cuda.device_count()==1; name=torch.cuda.get_device_name(0); "
                     "assert '3090' in name, name; print(json.dumps({'cuda_devices':1,'gpu':name,'torch':str(torch.__version__)}))"])
        checkpoint = args.initial_checkpoint.resolve()
        accumulated = [path.resolve() for path in args.teacher_data]
        rounds = []
        for round_index in range(args.rounds):
            number = round_index + 1
            prefix = f"round_{number:02d}"
            self.update(round=number)
            directory = self.run_dir / prefix
            directory.mkdir()
            collection = directory / "dagger"
            train_seeds = [args.seeds + round_index * 100 + i for i in range(3)]
            self.command(prefix + "_collect", ["-m", "foundation.rollout", "collect", "--output", collection,
                         "--modes", "vision", "--scene", "room", "--task-set", "legacy", "--seeds", *train_seeds,
                         "--spawn-swaps", 0, 1, "--duration", 20, "--spawn-jitter", .15, "--yaw-jitter", .12,
                         "--student-checkpoint", checkpoint, "--teacher-probability", PROBABILITIES[min(round_index, 2)],
                         "--fire-labels", "ray-safe", "--device", "cpu"])
            collected = audit_data([collection])
            if collected["episode_count"] != 6:
                raise RuntimeError(f"Expected 6 completed collection files, found {collected['episode_count']}")
            accumulated.append(collection)
            training = directory / "training"
            self.command(prefix + "_train", ["-m", "foundation.train", "--data", *accumulated,
                         "--checkpoint", checkpoint, "--output", training, "--steps", args.train_steps,
                         "--seed", args.seeds + 30000 + round_index, "--save-every", min(500, args.train_steps),
                         "--device", "cuda:0", "--wrist-error-limits", .08, .05, "--positive-fraction", .02,
                         "--wrist-residual-scale", 8., "--randomize-remaining", "--randomize-elapsed"])
            checkpoint = training / f"step_{args.train_steps:06d}.pt"
            if not checkpoint.is_file():
                raise RuntimeError("Training completed without its expected final-step checkpoint")
            screening = directory / "screening"
            screening_seeds = [args.seeds + 10000 + round_index * 100 + i for i in range(2)]
            self.command(prefix + "_screen", ["-m", "foundation.rollout", "evaluate", "--output", screening,
                         "--checkpoint", checkpoint, "--scene", "room", "--task-set", "legacy",
                         "--seeds", *screening_seeds, "--spawn-swaps", 0, 1, "--duration", 20,
                         "--spawn-jitter", .15, "--yaw-jitter", .12, "--device", "cpu"])
            screening_gate = assess_promotion(json.loads((screening / "summary.json").read_text()),
                                             expected_seeds=screening_seeds, duration=20.,
                                             expected_policy_class="foundation.policy.Policy")
            record = {"round": number, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                      "screening": screening_gate, "collected_data": collected}
            atomic_json(directory / "screening_gate.json", screening_gate)
            if screening_gate["passed"]:
                final = directory / "acceptance"
                acceptance_seeds = [args.seeds + 20000 + round_index * 100 + i for i in range(6)]
                self.command(prefix + "_acceptance", ["-m", "foundation.rollout", "evaluate", "--output", final,
                             "--checkpoint", checkpoint, "--scene", "room", "--task-set", "legacy",
                             "--seeds", *acceptance_seeds, "--spawn-swaps", 0, 1, "--duration", 60,
                             "--spawn-jitter", .3, "--yaw-jitter", .2, "--device", "cpu"])
                gate = assess_promotion(json.loads((final / "summary.json").read_text()),
                                        expected_seeds=acceptance_seeds, duration=60.,
                                        expected_policy_class="foundation.policy.Policy")
                record["acceptance"] = gate
                atomic_json(directory / "acceptance_gate.json", gate)
                if gate["passed"]:
                    export = directory / "export"
                    self.command(prefix + "_export", ["-m", "foundation.export", "--checkpoint", checkpoint,
                                 "--output", export, "--data", *accumulated, "--samples", 64])
                    record["candidate_export"] = str(export)
                    onnx_final = directory / "onnx_acceptance"
                    self.command(prefix + "_onnx_acceptance", ["-m", "foundation.rollout", "evaluate",
                                 "--output", onnx_final, "--onnx-model", export / "models/foundation.onnx",
                                 "--onnx-metadata", export / "models/metadata.json", "--scene", "room",
                                 "--task-set", "legacy", "--seeds", *acceptance_seeds, "--spawn-swaps", 0, 1,
                                 "--duration", 60, "--spawn-jitter", .3, "--yaw-jitter", .2, "--device", "cpu"])
                    onnx_gate = assess_promotion(json.loads((onnx_final / "summary.json").read_text()),
                                                expected_seeds=acceptance_seeds, duration=60.,
                                                expected_policy_class="foundation.onnx_policy.Policy")
                    record["onnx_acceptance"] = onnx_gate
                    atomic_json(directory / "onnx_acceptance_gate.json", onnx_gate)
                    if both_backends_pass(gate, onnx_gate):
                        rounds.append(record)
                        promoted = {"status": "promoted_room_baseline", "scope": SCOPE, "checkpoint": str(checkpoint),
                                    "checkpoint_sha256": digest(checkpoint), "submission": str(export / "submission.zip"),
                                    "submission_sha256": digest(export / "submission.zip"), "rounds": rounds,
                                    "closed_loop_backends_passed": ["torch", "onnxruntime_cpu"],
                                    "city_acceptance": "not_tested", "completed_at": now()}
                        atomic_json(self.run_dir / "promoted.json", promoted)
                        self.update(status="completed", stage="room_baseline_promoted", accepted=True,
                                    checkpoint=str(checkpoint), submission=promoted["submission"])
                        return 0
                    atomic_json(export / "not_promoted.json", {"status": "candidate_only", "scope": SCOPE,
                                "reason": "Exported ONNX failed the actual closed-loop gate", "gate": onnx_gate})
            rounds.append(record)
            atomic_json(directory / "result.json", record)
            self.update(stage="round_unmet", checkpoint=str(checkpoint),
                        latest_gate=record.get("onnx_acceptance", record.get("acceptance", screening_gate)))
        atomic_json(self.run_dir / "unmet.json", {"status": "unmet", "scope": SCOPE, "rounds": rounds,
                    "last_candidate": str(checkpoint), "accepted_checkpoint": None,
                    "reason": "Bounded iteration budget exhausted without satisfying the closed-loop gate", "completed_at": now()})
        self.update(status="unmet", stage="bounded_rounds_finished", accepted=False, checkpoint=str(checkpoint))
        return 2


def self_test():
    """Fast acceptance regressions, without Torch, assets, a GPU, or simulator."""
    episodes = [{"config": {"scene": "room", "task_set": "legacy", "seed": seed,
                 "spawn_swap": bool(swap), "duration": 60.}, "mode": "student",
                 "observation_source": "public_only", "cameras": True, "opponent_nonvisual": True,
                 "full_episode": True, "sim_seconds": 60., "score": 10., "falls": 0, "deaths": 0,
                 "invalid_actions": 0, "resets": 0, "health_lost": 0.}
                for seed in (1, 2, 3) for swap in (0, 1)]
    def passed(values):
        return assess_promotion({"episodes": values}, expected_seeds=[1, 2, 3], duration=60.)["passed"]
    assert passed(episodes)
    zero_scores = copy.deepcopy(episodes)
    for episode in zero_scores:
        episode["score"] = 0.
    assert not passed(zero_scores)
    for key, value in (("falls", 1), ("deaths", 1), ("invalid_actions", 1), ("resets", 1),
                       ("full_episode", False), ("sim_seconds", 10.), ("sim_seconds", float("nan")),
                       ("health_lost", 31.), ("score", float("nan"))):
        changed = copy.deepcopy(episodes)
        changed[0][key] = value
        assert not passed(changed), key
    assert not passed(episodes[:-1])
    duplicate = copy.deepcopy(episodes)
    duplicate[0] = duplicate[1]
    assert not passed(duplicate)
    assert not passed([])
    torch_gate = {"passed": True, "expected_policy_class": "foundation.policy.Policy"}
    onnx_gate = {"passed": True, "expected_policy_class": "foundation.onnx_policy.Policy"}
    assert both_backends_pass(torch_gate, onnx_gate)
    assert not both_backends_pass(torch_gate, {**onnx_gate, "passed": False})
    assert not both_backends_pass({**torch_gate, "passed": False}, onnx_gate)
    assert not both_backends_pass(torch_gate, None)
    assert not both_backends_pass(torch_gate, torch_gate)
    backend_check = assess_promotion({"episodes": episodes}, expected_seeds=[1, 2, 3], duration=60.,
                                    expected_policy_class="foundation.onnx_policy.Policy")
    assert not backend_check["passed"]  # No ONNX provenance in the fixture.
    print(json.dumps({"self_test": "passed", "cases": 20,
                      "checks": "score/safety/duration/evidence gates; both real inference backends and backend provenance required"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--teacher-data", type=Path, nargs="+")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--seeds", type=int, default=9000, help="Base seed for disjoint train, screening, and acceptance partitions")
    parser.add_argument("--max-wall-seconds", type=float, default=10800.)
    parser.add_argument("--stage-timeout", type=float, default=3600.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.run_dir or not args.initial_checkpoint or not args.teacher_data:
        parser.error("--run-dir, --initial-checkpoint, and --teacher-data are required")
    if not 1 <= args.rounds <= 10 or not 1 <= args.train_steps <= 50000 or args.seeds < 0:
        parser.error("rounds must be1..10, train-steps1..50000, and seeds nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in (args.max_wall_seconds, args.stage_timeout)):
        parser.error("Time limits must be finite and positive")
    if not args.initial_checkpoint.is_file():
        parser.error("Initial checkpoint does not exist")
    if args.run_dir.exists():
        parser.error("run-dir must be a new path; existing runs are never overwritten")
    data_audit = audit_data(args.teacher_data)
    reserved = {args.seeds + offset + r * 100 + i for offset in (10000, 20000)
                for r in range(args.rounds) for i in range(6)}
    if reserved.intersection(data_audit["seeds"]):
        parser.error("Held-out evaluation seeds overlap supplied training data; choose another --seeds base")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    driver = Driver(args, data_audit)
    if os.name == "posix":
        def handle_termination(signum, frame):
            raise KeyboardInterrupt(f"Driver received signal {signum}")
        signal.signal(signal.SIGTERM, handle_termination)
    try:
        return driver.run()
    except BaseException as error:
        driver.terminate_owned_child()
        record = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "stage": driver.state["stage"], "accepted": False, "completed_at": now()}
        atomic_json(args.run_dir / "failed.json", record)
        driver.update(status="failed", error=str(error), child_pid=None)
        print(json.dumps(record), file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 124 if isinstance(error, TimeoutError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
