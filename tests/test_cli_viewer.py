"""Exercise mode selection and the live/replay loops without opening a window."""
import argparse
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import sys

import mujoco
import pytest

from embodied_arena import cli


@pytest.mark.parametrize("command", ["run", "replay"])
@pytest.mark.parametrize("mode", [None, "simple", "debug"])
def test_cli_render_mode_selection(monkeypatch, command, mode):
    argv = ["embodied-arena", command]
    if command == "replay":
        argv.append("saved-match")
    if mode is not None:
        argv.extend(["--render-mode", mode])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, command, lambda args: args)
    args = cli.main()
    assert args.render_mode == (mode or "simple")
    if command == "run":
        assert not args.viewer


@pytest.mark.parametrize("mode", [None, "simple", "debug"])
def test_live_and_replay_share_mode_and_cleanup(monkeypatch, tmp_path, mode):
    windows = []

    class Viewer:
        def __init__(self, data):
            self.data = data
            self.frames = []
            self.texts = []
            self.closed = False

        def is_running(self):
            return True

        def set_texts(self, texts):
            self.texts.append(texts)

        def sync(self):
            self.frames.append(self.data.qpos.copy())

    @contextmanager
    def simple(model, data, *, camera):
        viewer = Viewer(data)
        viewer.mode = "simple"
        viewer.camera = camera
        windows.append(viewer)
        try:
            yield viewer
        finally:
            viewer.closed = True

    @contextmanager
    def debug(model, data, *, key_callback):
        viewer = Viewer(data)
        viewer.mode = "debug"
        viewer.cam = mujoco.MjvCamera()
        viewer.opt = mujoco.MjvOption()
        viewer.lock = nullcontext
        windows.append(viewer)
        try:
            yield viewer
        finally:
            viewer.closed = True

    monkeypatch.setattr(cli, "simple_viewer", simple)
    monkeypatch.setattr(cli, "passive_viewer", debug)
    monkeypatch.setattr(cli, "load_policy", lambda *args: SimpleNamespace(
        act=lambda obs: {"joint_targets": obs["proprio"]["q"], "fire": False}))
    args = argparse.Namespace(
        config=None, duration=.04, seed=7, scene="room", output=tmp_path,
        policy_a="unused", policy_b="unused", camera="overview", inline=True,
        viewer=True, realtime=False, no_cameras=True, no_preview=True,
        gif=False, timeout=1.0,
    )
    replay_args = argparse.Namespace(directory=tmp_path, speed=100., camera="overview")
    # Older local wrappers construct Namespaces without the new field.
    if mode is not None:
        args.render_mode = replay_args.render_mode = mode
    cli.run(args)
    cli.replay(replay_args)

    assert len(windows) == 2
    for viewer in windows:
        assert viewer.mode == (mode or "simple")
        assert viewer.closed
        # The caller-thread viewer may redraw a frame while pacing replay.
        assert len(viewer.frames) >= 2
        assert len(viewer.texts) == 2
        if viewer.mode == "simple":
            assert viewer.camera == "overview"
        else:
            assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
            assert viewer.cam.fixedcamid >= 0
            assert viewer.opt.geomgroup.tolist() == [1, 0, 0, 1, 1, 0]
            assert not viewer.opt.sitegroup.any()
    assert "REPLAY" in windows[1].texts[-1][2]


def test_slow_playback_processes_close_without_waiting_for_next_frame(monkeypatch):
    now = [10.0]
    sleeps = []
    viewer = SimpleNamespace(running=True, draws=0)

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    def sync():
        viewer.draws += 1
        # Simulate GLFW delivering a close event during a long frame wait.
        if viewer.draws == 3:
            viewer.running = False

    viewer.is_running = lambda: viewer.running
    viewer.sync = sync
    monkeypatch.setattr(cli.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert not cli._wait_until(viewer, "simple", deadline=12.0)
    assert viewer.draws == 3
    assert now[0] == pytest.approx(10.05)
    assert all(0 < delay <= 1 / 60 for delay in sleeps)


def test_playback_wait_honors_deadline_and_render_time(monkeypatch):
    now = [0.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    def sync():
        now[0] += .005

    viewer = SimpleNamespace(is_running=lambda: True, sync=sync)
    monkeypatch.setattr(cli.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert cli._wait_until(viewer, "simple", deadline=2.0)
    assert 2.0 <= now[0] <= 2.005
    assert len(sleeps) > 1
    assert all(0 < delay <= 1 / 60 for delay in sleeps)
