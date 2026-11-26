"""Competition adapter: deterministic tensor preprocessing and neural inference.

There is no teacher, target detector, inverse kinematics, or firing heuristic in
the deployed policy. Torch is imported lazily so collectors can use the encoder.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_calibration():
    calibration = json.loads((ROOT / "configs/g1_actuators.json").read_text(encoding="utf-8"))
    robot = ET.parse(ROOT / "third_party/g1/g1.xml").getroot()
    ranges = {j.attrib["name"]: [float(x) for x in j.attrib["range"].split()]
              for j in robot.findall(".//joint") if "name" in j.attrib and "range" in j.attrib}
    calibration["joint_limits"] = [ranges[name] for name in calibration["dof_names"]]
    return calibration


def projected_gravity(quaternion):
    w, x, y, z = quaternion
    return np.asarray([2 * (w*y - z*x), -2 * (z*y + w*x), 1 - 2 * (w*w + z*z)])


class ObservationEncoder:
    """Public observation -> fixed tensors; history preserves the raw gait output."""
    def __init__(self, calibration=None):
        self.calibration = copy.deepcopy(calibration or load_calibration())
        self.joint_names = list(self.calibration.get("dof_names", self.calibration.get("joint_names", [])))
        self.home = np.asarray(self.calibration["home"], dtype=np.float32)
        self.action_scale = float(self.calibration.get("action_scale", 0.25))
        self.limits = np.asarray(self.calibration.get("joint_limits", [[-np.inf, np.inf]] * 29), dtype=np.float32)
        if len(self.joint_names) != 29 or self.home.shape != (29,):
            raise ValueError("Expected full 29-DoF G1 calibration")
        self.reset()

    def reset(self, reset_count=None):
        self.reset_count = reset_count
        self.last = np.zeros(29, dtype=np.float32)
        self.phase = np.asarray([0., np.pi], dtype=np.float64)
        self.standing = False
        self.last_visual_time = None
        self.cached_image = np.zeros((4, 48, 64), dtype=np.float32)
        self.cached_image[3] = 1.
        self.reference_cache = {}

    def targets_to_action(self, targets, observed_joint_names):
        order = [observed_joint_names.index(name) for name in self.joint_names]
        canonical = np.asarray(targets, dtype=np.float32)[order]
        canonical = np.clip(canonical, self.limits[:, 0], self.limits[:, 1])
        return ((canonical - self.home) / self.action_scale).astype(np.float32)

    def action_to_targets(self, action, observed_joint_names, clip=True):
        canonical = self.home + self.action_scale * np.asarray(action, dtype=np.float32)
        if clip:
            canonical = np.clip(canonical, self.limits[:, 0], self.limits[:, 1])
        order = [self.joint_names.index(name) for name in observed_joint_names]
        return canonical[order].astype(np.float32)

    def encode(self, observation):
        if observation["reset_count"] != self.reset_count:
            self.reset(observation["reset_count"])
        p = observation["proprio"]
        order = [p["joint_names"].index(name) for name in self.joint_names]
        terms = {
            "actions": self.last, "base_ang_vel": np.asarray(p["gyro"]) * .25,
            "command_ang_vel": [0.], "command_lin_vel": [0., 0.],
            "cos_phase": np.cos(self.phase), "dof_pos": np.asarray(p["q"])[order] - self.home,
            "dof_vel": np.asarray(p["dq"])[order] * .05,
            "projected_gravity": projected_gravity(p["base_quat"]), "sin_phase": np.sin(self.phase),
        }
        proprio = np.concatenate([terms[key] for key in sorted(terms)]).astype(np.float32)
        visual = observation.get("visual")
        if visual is not None and visual["time"] != self.last_visual_time:
            rgb = np.asarray(Image.fromarray(np.asarray(visual["rgb"], dtype=np.uint8)).resize(
                (64, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 127.5 - 1.
            depth = np.nan_to_num(np.asarray(visual["depth"], dtype=np.float32), nan=10., posinf=10., neginf=0.)
            depth = np.asarray(Image.fromarray(depth).resize((64, 48), Image.Resampling.NEAREST))
            depth = np.log1p(np.clip(depth, 0., 10.)) / np.log(11.)
            self.cached_image = np.concatenate([rgb.transpose(2, 0, 1), depth[None]], axis=0).astype(np.float32)
            self.last_visual_time = visual["time"]
        references = np.zeros((20, 3, 24, 24), dtype=np.float32)
        mask = np.zeros(20, dtype=np.float32)
        # Preserve public task order; visual references carry target appearance.
        for index, task in enumerate(observation.get("tasks", [])[:20]):
            key = str(task["id"])
            if key not in self.reference_cache:
                rgb = np.asarray(Image.fromarray(np.asarray(task["reference_rgb"], dtype=np.uint8)).resize(
                    (24, 24), Image.Resampling.BILINEAR), dtype=np.float32) / 127.5 - 1.
                self.reference_cache[key] = rgb.transpose(2, 0, 1)
            references[index] = self.reference_cache[key]
            mask[index] = 1.
        game = observation["game"]
        game_vector = np.asarray([
            game["health"] / 100., game["score"] / 100., float(game["alive"]),
            min(game["stun_remaining"], 10.) / 10., min(game["cooldown_remaining"], 10.) / 10.,
            game["remaining"] / 60., observation["time"] / 60.,
            min(10., max(0., observation["time"] - visual["time"]) / .1) if visual is not None else 10.,
        ], dtype=np.float32)
        return {"proprio": proprio, "image": self.cached_image.copy(), "tasks": references,
                "task_mask": mask, "game": game_vector}

    def commit(self, action, command):
        """Commit the raw pretrained gait action, before learned arm residuals."""
        self.last = np.asarray(action, dtype=np.float32).copy()
        command = np.asarray(command)
        self.phase = (self.phase + 2*np.pi*.02 + np.pi) % (2*np.pi) - np.pi
        if np.linalg.norm(command[:2]) < .01 and abs(command[2]) < .01:
            self.phase[:] = np.pi
            self.standing = True
        elif self.standing:
            self.phase[:] = [0., np.pi]
            self.standing = False


class Policy:
    def __init__(self, player_id=0, config=None, checkpoint=None, device="cpu"):
        import torch
        from .model import FoundationNet
        self.torch = torch
        self.player_id = player_id
        self.device = torch.device(device)
        torch.set_num_threads(1)
        config = config or {}
        checkpoint = checkpoint or config.get("checkpoint") or os.environ.get("FOUNDATION_CHECKPOINT") or ROOT / "models/foundation.pt"
        self.model = FoundationNet.from_checkpoint(checkpoint, map_location=self.device).eval()
        self.encoder = ObservationEncoder(self.model.calibration)
        self.last_prediction = None

    def act(self, observation):
        encoded = self.encoder.encode(observation)
        batch = {name: self.torch.from_numpy(value).unsqueeze(0).to(self.device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            predicted = self.model(batch)
        self.last_prediction = {name: value[0].detach().cpu().numpy().copy() for name, value in predicted.items()}
        targets = self.encoder.action_to_targets(self.last_prediction["action"], observation["proprio"]["joint_names"])
        self.encoder.commit(self.last_prediction["gait_action"], self.last_prediction["command"])
        return {"joint_targets": targets, "fire": bool(self.last_prediction["fire_logit"] > 0.)}
