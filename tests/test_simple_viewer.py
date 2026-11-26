"""Spectator camera and UI behavior without opening an OpenGL window."""
import ctypes
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from embodied_arena.simple_viewer import (
    SimpleViewer, ViewController, button_at, button_layout,
)


@pytest.fixture
def world():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <camera name="overview" pos="0 -10 8"/>
          <camera name="city_hero" pos="5 -8 6"/>
          <body name="p0/pelvis" pos="-1 0 .8">
            <freejoint/><geom type="sphere" size=".1"/>
            <camera name="p0/eye" pos=".12 0 .3" xyaxes="0 -1 0 0 0 1"/>
          </body>
          <body name="p1/pelvis" pos="1 0 .8">
            <freejoint/><geom type="sphere" size=".1"/>
            <camera name="p1/eye" pos=".12 0 .3" xyaxes="0 -1 0 0 0 1"/>
          </body>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def rendered_camera(model, data, controller):
    scene = mujoco.MjvScene(model, maxgeom=100)
    mujoco.mjv_updateScene(model, data, controller.opt, None, controller.cam,
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    return (np.mean([camera.pos for camera in scene.camera], axis=0),
            np.mean([camera.forward for camera in scene.camera], axis=0))


@pytest.mark.parametrize("pid", [0, 1])
def test_player_view_uses_actual_eye_camera(world, pid):
    model, data = world
    controller = ViewController(model, data)
    controller.select(f"eye_p{pid + 1}")
    eye = model.camera(f"p{pid}/eye").id
    assert controller.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert controller.cam.fixedcamid == eye
    position, forward = rendered_camera(model, data, controller)
    np.testing.assert_allclose(position, data.cam_xpos[eye], atol=1e-6)
    np.testing.assert_allclose(forward, -data.cam_xmat[eye].reshape(3, 3)[:, 2])
    assert controller.opt.geomgroup.tolist() == [1, 0, 0, 1, 1, 1]
    assert not controller.opt.sitegroup.any()


@pytest.mark.parametrize("camera,walls", [("overview", 0), ("city_hero", 1)])
def test_global_button_restores_requested_initial_camera(world, camera, walls):
    model, data = world
    controller = ViewController(model, data, camera=camera)
    assert controller.mode == "global"
    controller.select("follow_p2")
    controller.select("eye_p1")
    controller.select("global")
    assert controller.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert controller.cam.fixedcamid == model.camera(camera).id
    assert controller.opt.geomgroup[5] == walls


@pytest.mark.parametrize("pid", [0, 1])
def test_follow_stays_behind_selected_player_after_translation_and_turn(world, pid):
    model, data = world
    controller = ViewController(model, data)
    controller.select(f"follow_p{pid + 1}")
    body = model.body(f"p{pid}/pelvis").id
    q = model.jnt_qposadr[model.body_jntadr[body]]
    other_body = model.body(f"p{1 - pid}/pelvis").id
    other_position = data.xpos[other_body].copy()
    initial_offset = controller.cam.lookat - data.xpos[body]
    for position, yaw in [([-1, 2, .8], 0), ([2, -3, 1.1], math.pi / 2),
                          ([0, 0, .3], -math.pi * .9)]:
        data.qpos[q:q + 3] = position
        data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
        controller.update()
        np.testing.assert_allclose(controller.cam.lookat, data.xpos[body] + initial_offset)
        camera_position, camera_forward = rendered_camera(model, data, controller)
        robot_forward = data.xmat[body].reshape(3, 3)[:, 0]
        # Check MuJoCo's resulting camera, not merely the azimuth formula.
        assert np.dot((camera_position - controller.cam.lookat)[:2], robot_forward[:2]) < -1
        assert np.dot(camera_forward[:2], robot_forward[:2]) > .8
        assert camera_position[2] > controller.cam.lookat[2]
        np.testing.assert_allclose(data.xpos[other_body], other_position)
    assert controller.opt.geomgroup[5] == 1


def test_switching_spectator_views_does_not_change_simulation_or_policy_cameras(world):
    model, data = world
    controller = ViewController(model, data)
    original = [array.copy() for array in (model.cam_pos, model.cam_quat,
                                           data.qpos, data.qvel, data.cam_xpos, data.cam_xmat)]
    for mode in ("eye_p1", "eye_p2", "follow_p1", "follow_p2", "global"):
        controller.select(mode)
        rendered_camera(model, data, controller)
    for actual, expected in zip((model.cam_pos, model.cam_quat, data.qpos,
                                 data.qvel, data.cam_xpos, data.cam_xmat), original):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("obstacle_group", [0, 3, 5])
def test_follow_moves_in_front_of_obstruction_and_recovers_when_it_moves(obstacle_group):
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco><worldbody>
          <camera name="overview" pos="0 -10 8"/>
          <body name="p0/pelvis" pos="0 0 .8">
            <freejoint/><geom type="sphere" size=".1" group="1"/>
            <camera name="p0/eye"/>
          </body>
          <body name="p1/pelvis" pos="10 0 .8">
            <freejoint/><geom type="sphere" size=".1" group="2"/>
            <camera name="p1/eye"/>
          </body>
          <body name="obstacle" mocap="true" pos="-1.5 5 2">
            <geom type="box" size=".1 1 2" group="{obstacle_group}"/>
          </body>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = ViewController(model, data)
    controller.select("follow_p1")
    full_distance = controller.cam.distance
    full_position, _ = rendered_camera(model, data, controller)
    assert full_position[0] < -1.6

    # Move a real geometric obstacle into the sightline, with no ray mocking.
    data.mocap_pos[0, 1] = 0
    mujoco.mj_forward(model, data)
    controller.update()
    position, _ = rendered_camera(model, data, controller)
    assert 0 < controller.cam.distance < full_distance
    assert -1.4 < position[0] < 0  # Camera is on the player's side of the wall.

    data.mocap_pos[0, 1] = 5
    mujoco.mj_forward(model, data)
    controller.update()
    assert controller.cam.distance == pytest.approx(full_distance)

    # Even an obstacle closer than the usual safety margin must stay behind
    # the camera, rather than forcing a fixed minimum through its surface.
    data.mocap_pos[0] = [-.16, 0, 2]
    mujoco.mj_forward(model, data)
    controller.update()
    close_position, _ = rendered_camera(model, data, controller)
    assert 0 < controller.cam.distance < .06
    assert -.06 < close_position[0] < 0
    data.mocap_pos[0, 1] = 5
    mujoco.mj_forward(model, data)
    controller.update()
    np.testing.assert_allclose(rendered_camera(model, data, controller)[0], full_position)

    # A wall past the desired camera position must not pull it closer.
    data.mocap_pos[0] = [-6, 0, 2]
    mujoco.mj_forward(model, data)
    controller.update()
    assert controller.cam.distance == pytest.approx(full_distance)


@pytest.mark.parametrize("robot_group", [1, 2, 4])
def test_follow_ignores_robot_parts_even_when_in_the_camera_sightline(robot_group):
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco><worldbody>
          <camera name="overview" pos="0 -10 8"/>
          <body name="p0/pelvis" pos="0 0 .8">
            <freejoint/><geom type="sphere" size=".1" group="1"/>
            <camera name="p0/eye"/>
            <body name="p0/arm" pos="-.8 0 .7">
              <geom name="arm" type="sphere" size=".3" group="{robot_group}"/>
            </body>
          </body>
          <body name="p1/pelvis" pos="10 0 .8">
            <freejoint/><geom type="sphere" size=".1" group="2"/>
            <camera name="p1/eye"/>
          </body>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = ViewController(model, data)
    controller.select("follow_p1")
    full_distance = controller.cam.distance
    position, _ = rendered_camera(model, data, controller)
    assert full_distance > 3 and position[0] < -3

    # The same shape does block when classified as scenery, proving that the
    # assertion above exercises filtering rather than a ray that misses it.
    model.geom_group[model.geom("arm").id] = 0
    controller.update()
    assert controller.cam.distance < 1
    model.geom_group[model.geom("arm").id] = robot_group
    controller.update()
    assert controller.cam.distance == pytest.approx(full_distance)


@pytest.mark.parametrize("size,framebuffer", [
    ((1280, 800), (1280, 800)), ((640, 360), (640, 360)),
    ((960, 540), (1920, 1080)), ((1280, 720), (1536, 864)),
])
def test_click_targets_match_rendered_buttons_after_resize_and_dpi_scaling(size, framebuffer):
    width, height = size
    fb_width, fb_height = framebuffer
    buttons = button_layout(fb_width, fb_height, fb_height / height)
    assert [mode for mode, _, _ in buttons] == [
        "eye_p1", "eye_p2", "global", "follow_p1", "follow_p2",
    ]
    previous_right = 0
    for mode, _, rect in buttons:
        assert rect.left >= previous_right
        assert 0 <= rect.bottom < rect.bottom + rect.height <= fb_height
        previous_right = rect.left + rect.width
        assert previous_right <= fb_width
        x = (rect.left + rect.width / 2) * width / fb_width
        y = (fb_height - rect.bottom - rect.height / 2) * height / fb_height
        assert button_at(x, y, width, height, fb_width, fb_height) == mode
    assert button_at(width / 2, height / 2, width, height, fb_width, fb_height) is None
    assert button_at(0, 0, width, height, fb_width, fb_height) is None


@pytest.mark.parametrize("dimensions", [(0, 800, 1280, 800), (1280, 0, 1280, 800),
                                         (1280, 800, 0, 800), (1280, 800, 1280, 0)])
def test_minimized_window_does_not_produce_click_target(dimensions):
    assert button_at(10, 10, *dimensions) is None


@pytest.mark.parametrize("previous_value", [41, 42])
def test_close_frees_resources_in_own_context_and_preserves_other_renderers(monkeypatch, previous_value):
    import embodied_arena.simple_viewer as module
    viewer = SimpleViewer.__new__(SimpleViewer)
    viewer.window = ctypes.c_void_p(42)
    viewer.scene = object()
    current = [ctypes.c_void_p(previous_value)]
    freed = []
    destroyed = []
    viewer.context = SimpleNamespace(free=lambda: freed.append(current[0].value))
    monkeypatch.setattr(module.glfw, "get_current_context", lambda: current[0])
    monkeypatch.setattr(module.glfw, "make_context_current", lambda context: current.__setitem__(0, context))
    monkeypatch.setattr(module.glfw, "destroy_window", lambda window: destroyed.append(window.value))
    monkeypatch.setattr(module.glfw, "terminate", lambda: pytest.fail("Must not terminate shared GLFW"))
    viewer.close()
    viewer.close()
    assert freed == [42] and destroyed == [42]
    assert viewer.window is None and viewer.context is None
    assert (current[0].value if current[0] else None) == (41 if previous_value == 41 else None)


def test_render_failure_restores_arena_gl_context(monkeypatch, world):
    import embodied_arena.simple_viewer as module
    model, data = world
    viewer = SimpleViewer.__new__(SimpleViewer)
    viewer.window = ctypes.c_void_p(42)
    viewer.model, viewer.data = model, data
    viewer.controller = ViewController(model, data)
    viewer.scene = object()
    viewer._font_scale = 150
    current = [ctypes.c_void_p(41)]
    monkeypatch.setattr(module.glfw, "get_current_context", lambda: current[0])
    monkeypatch.setattr(module.glfw, "make_context_current", lambda context: current.__setitem__(0, context))
    monkeypatch.setattr(module.glfw, "window_should_close", lambda window: False)
    monkeypatch.setattr(module.glfw, "poll_events", lambda: None)
    monkeypatch.setattr(module.glfw, "get_framebuffer_size", lambda window: (1280, 800))
    monkeypatch.setattr(module.glfw, "get_window_size", lambda window: (1280, 800))
    def fail(*args):
        assert current[0].value == 42
        raise RuntimeError("draw failed")
    monkeypatch.setattr(module.mujoco, "mjv_updateScene", fail)
    with pytest.raises(RuntimeError, match="draw failed"):
        viewer.sync()
    assert current[0].value == 41
