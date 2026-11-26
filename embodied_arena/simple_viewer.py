"""Compact spectator window using MuJoCo's renderer and GLFW.

Rendering runs on the caller's thread. Every draw restores the previous OpenGL
context so the arena's RGB-D renderers can keep using their own contexts.
"""
from contextlib import contextmanager
import ctypes
import math

import glfw
import mujoco
import numpy as np


VIEWS = (
    ("eye_p1", "1  P1 VIEW"),
    ("eye_p2", "2  P2 VIEW"),
    ("global", "3  GLOBAL"),
    ("follow_p1", "4  FOLLOW P1"),
    ("follow_p2", "5  FOLLOW P2"),
)
TOOLBAR_HEIGHT = 60


def button_layout(width, height, scale=1.0):
    """Return button rectangles in framebuffer coordinates (bottom origin)."""
    gap = max(1, round(8 * scale))
    margin = max(1, round(12 * scale))
    available = max(5, width - 2 * margin - gap * 4)
    top = height - round(10 * scale)
    button_height = max(1, round(40 * scale))
    return [(mode, label, mujoco.MjrRect(
        margin + round(i * available / 5) + i * gap,
        top - button_height,
        max(1, round((i + 1) * available / 5) - round(i * available / 5)),
        button_height,
    )) for i, (mode, label) in enumerate(VIEWS)]


def button_at(x, y, width, height, framebuffer_width, framebuffer_height):
    """Hit-test GLFW cursor coordinates, including Windows display scaling."""
    if min(width, height, framebuffer_width, framebuffer_height) <= 0:
        return None
    scale = framebuffer_height / height
    px, py = x * framebuffer_width / width, (height - y) * scale
    for mode, _, rect in button_layout(framebuffer_width, framebuffer_height, scale):
        if (rect.left <= px < rect.left + rect.width
                and rect.bottom <= py < rect.bottom + rect.height):
            return mode
    return None


class ViewController:
    """Camera selection without changing model cameras or policy observations."""

    def __init__(self, model, data, camera="overview"):
        self.model, self.data = model, data
        self.global_camera = camera
        self.global_id = model.camera(camera).id
        self.eyes = [model.camera(f"p{pid}/eye").id for pid in (0, 1)]
        self.bodies = [model.body(f"p{pid}/pelvis").id for pid in (0, 1)]
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 0, 0, 1, 1, 1]
        self.opt.sitegroup[:] = 0
        # Robot collision and visual groups must not shorten their own camera.
        self._obstacle_groups = np.array([1, 0, 0, 1, 0, 1], dtype=np.uint8)
        self._hit_geom = np.empty(1, dtype=np.int32)
        self.select("global")

    def select(self, mode):
        if mode not in {key for key, _ in VIEWS}:
            raise ValueError(f"Unknown spectator view: {mode}")
        self.mode = mode
        self.opt.geomgroup[5] = 0 if mode == "global" and self.global_camera in ("overview", "aerial") else 1
        self.cam.trackbodyid = -1
        if mode == "global":
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.global_id
        elif mode.startswith("eye_"):
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.eyes[0 if mode == "eye_p1" else 1]
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.fixedcamid = -1
            self.cam.distance = 3.5
            self.cam.elevation = -18
        self.update()

    def update(self):
        if not self.mode.startswith("follow_"):
            return
        body = self.bodies[0 if self.mode == "follow_p1" else 1]
        self.cam.lookat[:] = self.data.xpos[body]
        self.cam.lookat[2] += .45
        # Local +X is the robot's forward direction. Keep the camera behind it
        # while allowing the robot to turn, translate, fall and respawn.
        forward = self.data.xmat[body].reshape(3, 3)[:, 0]
        if math.hypot(forward[0], forward[1]) > 1e-6:
            self.cam.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
        yaw, pitch = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
        backward = np.array([-math.cos(yaw) * math.cos(pitch),
                             -math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
        obstacle = mujoco.mj_ray(self.model, self.data, self.cam.lookat, backward,
                                self._obstacle_groups, 1, body, self._hit_geom)
        # For very close surfaces, use half the available gap instead of a
        # fixed minimum that could put the camera on the far side of the wall.
        self.cam.distance = min(3.5, max(1e-6, obstacle * .5, obstacle - .2)) if obstacle >= 0 else 3.5


class SimpleViewer:
    def __init__(self, model, data, camera="overview"):
        self.model, self.data = model, data
        self.controller = ViewController(model, data, camera)
        self.window = self.context = self.scene = None
        self._texts = []
        self._font_scale = None
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW for the simple viewer")
        previous = glfw.get_current_context()
        try:
            # MuJoCo's offscreen renderer leaves the VISIBLE hint disabled.
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            self.window = glfw.create_window(1280, 800, "NEON ARENA | Simple Viewer", None, None)
            if not self.window:
                raise RuntimeError("Could not open the simple viewer window")
            glfw.set_window_size_limits(self.window, 640, 360, glfw.DONT_CARE, glfw.DONT_CARE)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.scene = mujoco.MjvScene(model, maxgeom=max(10000, model.ngeom + 1000))
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._font_scale = 150
            glfw.set_mouse_button_callback(self.window, self._mouse_button)
            glfw.set_key_callback(self.window, self._key)
        except BaseException:
            self.close()
            raise
        finally:
            glfw.make_context_current(previous)

    def _mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        mode = button_at(*glfw.get_cursor_pos(window), *glfw.get_window_size(window),
                         *glfw.get_framebuffer_size(window))
        if mode:
            self.controller.select(mode)

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if glfw.KEY_1 <= key <= glfw.KEY_5:
            self.controller.select(VIEWS[key - glfw.KEY_1][0])
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def is_running(self):
        return bool(self.window) and not glfw.window_should_close(self.window)

    def set_texts(self, texts):
        self._texts = list(texts) if isinstance(texts, list) else [texts]

    def sync(self):
        if not self.is_running():
            return
        glfw.poll_events()
        if not self.is_running():
            return
        width, height = glfw.get_framebuffer_size(self.window)
        window_width, window_height = glfw.get_window_size(self.window)
        if min(width, height, window_width, window_height) <= 0:
            return  # Minimized windows can have a zero-sized framebuffer.
        scale = height / window_height
        toolbar_height = round(TOOLBAR_HEIGHT * scale)
        viewport = mujoco.MjrRect(0, 0, width, max(1, height - toolbar_height))
        previous = glfw.get_current_context()
        try:
            glfw.make_context_current(self.window)
            font_scale = max(100, min(300, round(150 * scale / 50) * 50))
            if font_scale != self._font_scale:
                mujoco.mjr_changeFont(font_scale, self.context)
                self._font_scale = font_scale
            self.controller.update()
            mujoco.mjv_updateScene(self.model, self.data, self.controller.opt, None,
                                   self.controller.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
            mujoco.mjr_render(viewport, self.scene, self.context)
            for _, position, left, right in self._texts:
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_SHADOW,
                                   mujoco.mjtGridPos.mjGRID_TOPLEFT if position is None else position,
                                   viewport, left, right, self.context)
            mujoco.mjr_rectangle(mujoco.MjrRect(0, height - toolbar_height, width, toolbar_height),
                                 .055, .07, .10, 1)
            for mode, label, rect in button_layout(width, height, scale):
                color = (.08, .44, .55) if mode == self.controller.mode else (.16, .20, .26)
                mujoco.mjr_label(rect, mujoco.mjtFont.mjFONT_NORMAL, label,
                                 *color, 1, 1, 1, 1, self.context)
            glfw.swap_buffers(self.window)
        finally:
            glfw.make_context_current(previous)

    def close(self):
        if not self.window:
            return
        previous = glfw.get_current_context()
        if ctypes.cast(previous, ctypes.c_void_p).value == ctypes.cast(self.window, ctypes.c_void_p).value:
            previous = None
        try:
            glfw.make_context_current(self.window)
            if self.context is not None:
                self.context.free()
                self.context = None
            self.scene = None
        finally:
            glfw.destroy_window(self.window)
            self.window = None
            glfw.make_context_current(previous)
        # GLFW is also owned by MuJoCo's RGB-D renderers. Do not terminate it.


@contextmanager
def simple_viewer(model, data, camera="overview"):
    viewer = SimpleViewer(model, data, camera)
    try:
        yield viewer
    finally:
        viewer.close()
