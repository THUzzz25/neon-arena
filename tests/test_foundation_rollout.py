"""Meaningful collector checks without loading a policy checkpoint or renderer."""
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_arena.config import MatchConfig
from embodied_arena.policies import DiagnosticVision
from foundation.rollout import apply_spawn_jitter, consumed_command, dagger_teacher_act, load_episode, ray_safe_fire_label, run_episode, wrist_goal_label


def observation(time=0., health=100., resets=0):
    return {"time": time, "reset_count": resets,
            "proprio": {"joint_names": [str(i) for i in range(29)]},
            "visual": {"time": 0.}, "tasks": [{"id": "red", "kind": "public"}],
            "game": {"health": health}}


class FakeEncoder:
    def __init__(self):
        self.last = np.zeros(29, dtype=np.float32)
        self.phase = np.array([0., np.pi])
        self.standing = False

    def encode(self, obs):
        proprio = np.zeros(100, dtype=np.float32)
        proprio[:29] = self.last
        return {"proprio": proprio, "game": np.zeros(8, dtype=np.float32),
                "image": np.zeros((4, 48, 64), dtype=np.float32),
                "tasks": np.zeros((20, 3, 24, 24), dtype=np.float32),
                "task_mask": np.ones(20, dtype=np.float32)}

    def targets_to_action(self, targets, names):
        return np.asarray(targets, dtype=np.float32).copy()

    def commit(self, action, command):
        self.last = action.copy()


class FakePolicy:
    def __init__(self):
        self.command = np.zeros(3, dtype=np.float32)
        self.walk = SimpleNamespace(last=np.zeros(29, dtype=np.float32))

    def act(self, obs):
        self.command = self.command + .1  # next command, not consumed this tick
        self.walk.last = np.ones(29, dtype=np.float32) * .4
        return {"joint_targets": np.ones(29, dtype=np.float32) * .7, "fire": True}


class FakeArena:
    fail = False

    def __init__(self, config, cameras):
        self.config = config
        self.cameras = cameras
        self.time = 0.
        self.reset_count = 0
        self.robots = [SimpleNamespace(root_q=0, body=0, limits=np.tile([-1., 1.], (29, 1)))]
        self.data = SimpleNamespace(qpos=np.array([0., 0., .8]), xmat=np.eye(3)[None])
        self.referee = SimpleNamespace(events=[], players=[SimpleNamespace(health=100.)])
        self.shot_traces = []

    def observation(self, pid):
        return observation(self.time, self.referee.players[0].health, self.reset_count)

    def step(self, actions):
        self.time += .02
        self.shot_traces.append({"player": 0})
        self.referee.events.append({"type": "target_completed", "player": 0, "points": 10.})
        if self.fail:
            self.referee.events.append({"type": "fall", "player": 0})
            self.referee.events.append({"type": "fall_reset", "player": 0})
            self.reset_count += 1
            self.last_targets = [np.zeros(29)]
        return [10., 0.], self.time >= self.config.duration

    def close(self):
        pass


def test_command_label_is_pre_act_and_resets_to_zero():
    # Bypass the expensive constructor; only the public command lifecycle matters.
    policy = DiagnosticVision.__new__(DiagnosticVision)
    policy.command = [.18, 0., .2]
    policy.reset_count = 0
    before = consumed_command(policy, observation())
    policy.command[0] = 0.
    np.testing.assert_allclose(before, [.18, 0., .2])
    np.testing.assert_array_equal(consumed_command(policy, observation(resets=1)), np.zeros(3))


def test_collection_preserves_raw_gait_history_and_terminal_reward(tmp_path):
    path = tmp_path / "episode.npz"
    metrics = run_episode(MatchConfig(scene="room", duration=.04), FakePolicy(),
        opponent=FakePolicy(), arena_factory=FakeArena, encoder_factory=FakeEncoder, collect_path=path)
    arrays = load_episode(path, expand=True)
    assert metrics["score"] == 20. and metrics["full_episode"]
    assert arrays["action"].shape == (2, 29)
    assert arrays["wrist_goal"].shape == (2, 2)
    np.testing.assert_allclose(arrays["wrist_goal"], .7)
    np.testing.assert_allclose(arrays["action"], .7)
    np.testing.assert_allclose(arrays["proprio"][1, :29], .4)  # raw gait, not .7 wrist residual
    np.testing.assert_allclose(arrays["command"][0], 0.)
    np.testing.assert_allclose(arrays["command"][1], .1)
    np.testing.assert_array_equal(arrays["reward"], [10., 10.])
    assert arrays["image_frames"].shape[0] == arrays["task_frames"].shape[0] == 1
    assert arrays["image"].shape == (2, 4, 48, 64)
    assert arrays["tasks"].shape == (2, 20, 3, 24, 24)


def test_failure_stops_collection_and_does_not_replace_label_with_reset_home(tmp_path):
    class FallingArena(FakeArena):
        fail = True

    path = tmp_path / "failed.npz"
    metrics = run_episode(MatchConfig(scene="room", duration=60.), FakePolicy(),
        opponent=FakePolicy(), arena_factory=FallingArena, encoder_factory=FakeEncoder, collect_path=path)
    arrays = load_episode(path)
    assert metrics["control_steps"] == 1
    assert metrics["falls"] == metrics["resets"] == 1
    assert metrics["termination"] == "fall" and not metrics["full_episode"]
    np.testing.assert_allclose(arrays["action"][0], .7)
    assert arrays["failure"].tolist() == [True]


def test_health_restoration_never_counts_as_negative_damage():
    class HealingArena(FakeArena):
        def __init__(self, config, cameras):
            super().__init__(config, cameras)
            self.referee.players[0].health = 5.

        def step(self, actions):
            result = super().step(actions)
            self.referee.players[0].health = 100.
            return result

    metrics = run_episode(MatchConfig(scene="room", duration=.02), FakePolicy(),
                          opponent=FakePolicy(), arena_factory=HealingArena)
    assert metrics["health_lost"] == 0.


@pytest.mark.parametrize("beta,actual_gait,actual_action", [(0., .2, -.5), (1., .4, .7)])
def test_dagger_queries_teacher_from_executed_recurrence(tmp_path, beta, actual_gait, actual_action):
    class InspectingTeacher(FakePolicy):
        def __init__(self):
            super().__init__()
            self.seen_history = []

        def act(self, obs):
            self.seen_history.append(self.walk.last.copy())
            return super().act(obs)

    class FakeStudent:
        def __init__(self):
            self.encoder = FakeEncoder()
            self.last_prediction = None
            self.seen_history = []

        def act(self, obs):
            self.seen_history.append(self.encoder.last.copy())
            self.last_prediction = {"command": np.array([-.1, 0., 0.]), "gait_action": np.ones(29) * .2}
            self.encoder.commit(self.last_prediction["gait_action"], self.last_prediction["command"])
            return {"joint_targets": np.ones(29) * -.5, "fire": False}

    teacher, student = InspectingTeacher(), FakeStudent()
    path = tmp_path / "dagger.npz"
    run_episode(MatchConfig(scene="room", duration=.04), teacher,
        opponent=FakePolicy(), arena_factory=FakeArena, encoder_factory=FakeEncoder,
        collect_path=path, rollout_policy=student, teacher_probability=beta)
    arrays = load_episode(path)
    np.testing.assert_allclose(teacher.seen_history[1], actual_gait)
    np.testing.assert_allclose(student.seen_history[1], actual_gait)
    np.testing.assert_allclose(arrays["proprio"][1, :29], actual_gait)
    np.testing.assert_allclose(arrays["action"], .7)  # BC always targets queried teacher
    np.testing.assert_allclose(arrays["gait_action"], .4)
    np.testing.assert_allclose(arrays["executed_action"], actual_action)
    np.testing.assert_allclose(arrays["executed_gait_action"], actual_gait)
    np.testing.assert_allclose(arrays["teacher_used"], beta)


def test_dagger_wrist_origin_uses_measured_initial_then_executed_targets():
    class ArmTeacher:
        def __init__(self):
            self.arm = np.array([0, 1])
            self.robot = SimpleNamespace(names=[str(i) for i in range(29)])
            self.target_command = np.zeros(2)
            self.walk = SimpleNamespace(act=lambda: None)

        def act(self, obs):
            self.target_command[:] = 999.  # simulate DiagnosticVision reset
            self.walk.act()
            return self.target_command.copy()

    teacher = ArmTeacher()
    obs = observation()
    obs["proprio"]["q"] = np.arange(29, dtype=float) / 10.
    np.testing.assert_allclose(dagger_teacher_act(teacher, obs, None), [0., .1])
    np.testing.assert_allclose(dagger_teacher_act(teacher, obs, np.ones(29) * .7), [.7, .7])
    # The interception is temporary, including after a reset.
    assert teacher.walk.act() is None


def test_room_spawn_jitter_is_reproducible_preserves_height_and_does_not_count_as_reset():
    from embodied_arena.engine import Arena
    config = MatchConfig(scene="room", seed=918)
    first = Arena(config, cameras=False)
    second = Arena(config, cameras=False)
    try:
        old_height = [first.data.qpos[r.root_q + 2] for r in first.robots]
        a = apply_spawn_jitter(first, .3, .2)
        b = apply_spawn_jitter(second, .3, .2)
        assert a == b
        assert first.reset_counts == [0, 0]
        for index, robot in enumerate(first.robots):
            assert first.data.qpos[robot.root_q + 2] == old_height[index]
            np.testing.assert_allclose(np.linalg.norm(first.data.qpos[robot.root_q + 3:robot.root_q + 7]), 1.)
            assert abs(a["offsets"][index]["dx_m"]) <= .3
            assert abs(a["offsets"][index]["dy_m"]) <= .3
            assert abs(a["offsets"][index]["yaw_rad"]) <= .2
        assert any(abs(entry["dx_m"]) > 0. for entry in a["offsets"])
        np.testing.assert_array_equal(first.data.qvel, np.zeros(first.model.nv))
        with pytest.raises(ValueError, match="exactly once"):
            apply_spawn_jitter(first, .3, .2)
    finally:
        first.close()
        second.close()


def test_only_default_stationary_opponent_skips_camera(monkeypatch):
    import foundation.rollout as rollout
    observed = []

    class CameraArena(FakeArena):
        def observation(self, pid):
            observed.append((pid, self.cameras))
            return super().observation(pid)

    monkeypatch.setattr(rollout, "WalkInPlace", lambda **kwargs: FakePolicy())
    summary = run_episode(MatchConfig(scene="room", duration=.02), FakePolicy(), arena_factory=CameraArena)
    assert summary["opponent_nonvisual"]
    assert (1, False) in observed and (0, False) not in observed
    observed.clear()
    summary = run_episode(MatchConfig(scene="room", duration=.02), FakePolicy(), opponent=FakePolicy(), arena_factory=CameraArena)
    assert not summary["opponent_nonvisual"]
    assert all(cameras for pid, cameras in observed)


def test_ray_safe_probe_uses_actual_current_pose_and_preserves_rules_physics_and_traces():
    import mujoco
    from embodied_arena.engine import Arena
    arena = Arena(MatchConfig(scene="room"), cameras=False)
    try:
        robot = arena.robots[0]
        start = arena.data.site_xpos[robot.muzzle].copy()
        axis = arena.data.site_xmat[robot.muzzle].reshape(3, 3)[:, 0].copy()
        arena.shot_traces.append({"existing": True})
        for name, expected in (("red_cup", True), ("blue_cup", True), ("green_cup", False)):
            body = arena.model.body(name).id
            original_position = arena.model.body_pos[body].copy()
            arena.model.body_pos[body] = start + axis * .38
            mujoco.mj_forward(arena.model, arena.data)
            qpos, qvel, ctrl = arena.data.qpos.copy(), arena.data.qvel.copy(), arena.data.ctrl.copy()
            assert ray_safe_fire_label(arena) is expected
            np.testing.assert_array_equal(arena.data.qpos, qpos)
            np.testing.assert_array_equal(arena.data.qvel, qvel)
            np.testing.assert_array_equal(arena.data.ctrl, ctrl)
            assert arena.time == 0.
            assert arena.shot_traces == [{"existing": True}]
            assert arena.referee.events == []
            assert all(p.score == 0. and p.health == 100. for p in arena.referee.players)
            assert all(target.active for target in arena.referee.targets.values())
            arena.referee.targets[name].active = False
            assert ray_safe_fire_label(arena) is False
            arena.referee.targets[name].active = True
            arena.referee.players[0].next_shot = 1.
            assert ray_safe_fire_label(arena) is False
            assert arena.shot_traces == [{"existing": True}]
            arena.referee.players[0].next_shot = 0.
            arena.model.body_pos[body] = original_position
            mujoco.mj_forward(arena.model, arena.data)
    finally:
        arena.close()


def test_ray_safe_labels_do_not_replace_teacher_executed_fire(tmp_path, monkeypatch):
    import foundation.rollout as rollout
    monkeypatch.setattr(rollout, "ray_safe_fire_label", lambda arena: False)
    path = tmp_path / "ray_safe.npz"
    metrics = run_episode(MatchConfig(scene="room", duration=.04), FakePolicy(),
        opponent=FakePolicy(), arena_factory=FakeArena, encoder_factory=FakeEncoder,
        collect_path=path, fire_labels="ray-safe")
    arrays = load_episode(path)
    np.testing.assert_array_equal(arrays["fire"], [0., 0.])
    np.testing.assert_array_equal(arrays["teacher_fire_request"], [1., 1.])
    np.testing.assert_array_equal(arrays["executed_fire"], [1., 1.])
    assert metrics["shots"] == 2
    assert arrays["metadata"]["fire_label_source"] == "ray-safe"
    assert arrays["metadata"]["privileged_training_labels"]
    with pytest.raises(ValueError, match="training-only"):
        run_episode(MatchConfig(scene="room"), FakePolicy(), fire_labels="ray-safe")


def test_wrist_goal_uses_unsmoothed_public_ik_and_normalizes():
    policy = DiagnosticVision.__new__(DiagnosticVision)
    policy.aim_point = np.array([1., 0., 0.])
    policy.robot = SimpleNamespace(names=[str(i) for i in range(29)])
    policy.target = np.zeros(29)
    policy.target[-2:] = [.1, -.2]
    encoder = SimpleNamespace(targets_to_action=lambda target, names: np.asarray(target) / .25)
    fallback = np.ones(29) * .03
    np.testing.assert_allclose(wrist_goal_label(policy, encoder, fallback), [.4, -.8])
    policy.aim_point = None
    np.testing.assert_allclose(wrist_goal_label(policy, encoder, fallback), [.03, .03])


def test_onnx_evaluation_cli_uses_exported_cpu_policy_and_records_exact_hashes(tmp_path, monkeypatch):
    import hashlib
    import json
    import foundation.onnx_policy as onnx_policy
    import foundation.rollout as rollout
    model = tmp_path / "export.onnx"
    metadata = tmp_path / "metadata.json"
    model.write_bytes(b"exported-test-graph")
    metadata.write_text("{}", encoding="utf-8")
    constructed = []
    evaluated = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return "actual_onnx_adapter"

    def episode(config, policy, **kwargs):
        evaluated.append((policy, kwargs))
        return {"score": 10., "falls": 0, "misses": 0, "resets": 0, "sim_seconds": config.duration,
                "wall_seconds": 1., "termination": "duration", "full_episode": True, "no_fall": True,
                "traffic_collisions": 0, "deaths": 0, "invalid_actions": 0, "health_lost": 0.}

    monkeypatch.setattr(onnx_policy, "Policy", factory)
    monkeypatch.setattr(rollout, "run_episode", episode)
    output = tmp_path / "evaluation"
    rollout.main(["evaluate", "--onnx-model", str(model), "--onnx-metadata", str(metadata),
                  "--output", str(output), "--seeds", "1751", "--duration", "60"])
    assert constructed[0]["model_path"] == model
    assert constructed[0]["metadata_path"] == metadata
    assert "device" not in constructed[0]
    assert evaluated[0][0] == "actual_onnx_adapter"
    provenance = evaluated[0][1]["provenance"]
    assert provenance["inference_device"] == "cpu"
    assert provenance["onnx_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["episodes"][0]["onnx_metadata_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert summary["episodes"][0]["inference_backend"] == "onnxruntime"


@pytest.mark.parametrize("arguments", [
    ["evaluate", "--checkpoint", "model.pt", "--onnx-model", "model.onnx"],
    ["evaluate", "--onnx-metadata", "metadata.json"],
    ["collect", "--onnx-model", "model.onnx"],
    ["evaluate", "--onnx-model", "model.onnx", "--device", "cuda:0"],
])
def test_onnx_cli_rejects_ambiguous_or_unsupported_runtime(arguments):
    from foundation.rollout import main
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
