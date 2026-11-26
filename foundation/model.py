"""RGB-D/reference-conditioned network with a differentiable pretrained gait.

ONNX is needed only to initialize the first checkpoint. Inference checkpoints
contain the full gait, vision network, calibration, and architecture metadata.
"""
from __future__ import annotations

from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn


class GaitNetwork(nn.Module):
    def __init__(self, hidden_sizes=(512, 256, 128), eps=1e-5):
        super().__init__()
        layers = []
        width = 100
        for size in hidden_sizes:
            layers.extend([nn.Linear(width, size), nn.LayerNorm(size, eps=eps), nn.SiLU()])
            width = size
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Sequential(nn.Linear(width, 29))
        self.register_buffer("obs_mean", torch.zeros(1, 100))
        self.register_buffer("obs_divisor", torch.ones(1, 100))
        self.register_buffer("action_scale", torch.ones(29))
        self.register_buffer("action_bias", torch.zeros(29))

    def forward(self, proprio):
        x = (proprio - self.obs_mean) / self.obs_divisor
        return torch.tanh(self.fc_mu(self.net(x))) * self.action_scale + self.action_bias

    def import_onnx(self, path):
        import onnx
        from onnx import numpy_helper
        graph = onnx.load(str(path)).graph
        weights = {t.name: numpy_helper.to_array(t).copy() for t in graph.initializer}
        state = self.state_dict()
        for key in state:
            source = {"obs_mean": "obs_normalizer._mean", "obs_divisor": "onnx::Div_86"}.get(key, "actor." + key)
            if source not in weights:
                raise ValueError(f"Unsupported gait ONNX: missing {source}")
            value = torch.from_numpy(weights[source])
            if state[key].shape != value.shape:
                raise ValueError(f"Gait tensor shape mismatch: {source}")
            state[key] = value
        self.load_state_dict(state)


class FoundationNet(nn.Module):
    """One learned forward graph: public inputs -> joint actions and fire logit."""
    def __init__(self, onnx_path=None, *, gait_spec=None, residual_scale=0.12,
                 wrist_residual_scale=4.0, command_limits=(0.25, 0.10, 0.30),
                 freeze_gait=True, wrist_error_limits=None):
        super().__init__()
        self.model_config = {
            "gait_spec": copy.deepcopy(gait_spec or {"hidden_sizes": [512, 256, 128], "eps": 1e-5}),
            "residual_scale": float(residual_scale), "wrist_residual_scale": float(wrist_residual_scale),
            "command_limits": list(command_limits), "freeze_gait": bool(freeze_gait),
        }
        self.set_wrist_error_limits(wrist_error_limits)
        self.gait = GaitNetwork(**self.model_config["gait_spec"])
        if onnx_path is not None:
            self.gait.import_onnx(onnx_path)
        self.gait.requires_grad_(not freeze_gait)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(48 * 3 * 4, 128), nn.SiLU(),
        )
        self.task_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(96, 32), nn.SiLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(128 + 64 + 100 + 8, 256), nn.SiLU(),
                                    nn.Linear(256, 128), nn.SiLU())
        self.command_head = nn.Linear(128, 3)
        self.residual_head = nn.Linear(128, 29)
        self.fire_head = nn.Linear(128, 1)
        self.register_buffer("command_limits", torch.tensor(command_limits, dtype=torch.float32))
        scales = torch.full((29,), float(residual_scale))
        # Canonical calibration order: the last two joints are right wrist pitch/yaw.
        scales[-2:] = float(wrist_residual_scale)
        self.register_buffer("residual_limits", scales)
        nn.init.zeros_(self.command_head.weight)
        nn.init.zeros_(self.command_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.fire_head.weight)
        nn.init.constant_(self.fire_head.bias, -4.0)
        self.calibration = None
        self.checkpoint_metadata = {}

    def set_wrist_error_limits(self, limits):
        """Optionally bound wrist target error from measured q, in radians.

        This is a position-error bound, not a target slew-rate limiter. It adds
        no checkpoint tensors, so earlier checkpoints load without migration.
        """
        if limits is not None:
            limits = tuple(float(x) for x in limits)
            if len(limits) != 2 or not all(np.isfinite(x) and x > 0 for x in limits):
                raise ValueError("wrist_error_limits must contain two positive radian limits")
        self.wrist_error_limits = limits
        self.model_config["wrist_error_limits"] = None if limits is None else list(limits)

    def forward(self, batch):
        proprio = batch["proprio"]
        image = self.image_encoder(batch["image"])
        references = batch["tasks"]
        b, n = references.shape[:2]
        features = self.task_encoder(references.reshape(b * n, *references.shape[2:])).reshape(b, n, -1)
        mask = batch["task_mask"].to(features.dtype).unsqueeze(-1)
        mean = (features * mask).sum(1) / mask.sum(1).clamp_min(1)
        maximum = features.masked_fill(mask == 0, -1e4).amax(1)
        maximum = torch.where(mask.sum(1) > 0, maximum, torch.zeros_like(maximum))
        fused = self.fusion(torch.cat([image, mean, maximum, proprio, batch["game"]], dim=-1))
        command = torch.tanh(self.command_head(fused)) * self.command_limits
        # Alphabetical Holosoma terms: angular command index 32, linear 33:35.
        gait_input = torch.cat([proprio[:, :32], command[:, 2:3], command[:, :2], proprio[:, 35:]], dim=-1)
        gait_action = self.gait(gait_input)
        residual = torch.tanh(self.residual_head(fused)) * self.residual_limits
        action = gait_action + residual
        wrist_candidate = action[:, -2:]
        if self.wrist_error_limits is not None:
            # dof_pos = q - home; normalized action = (target - home) / .25.
            measured = proprio[:, 64:66] / .25
            limits = proprio.new_tensor(self.wrist_error_limits) / .25
            wrists = torch.maximum(measured - limits, torch.minimum(measured + limits, wrist_candidate))
            action = torch.cat([action[:, :-2], wrists], dim=-1)
        return {"action": action, "command": command,
                "fire_logit": self.fire_head(fused).squeeze(-1),
                "gait_action": gait_action, "residual": residual,
                "wrist_candidate": wrist_candidate}

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        checkpoint = torch.load(str(path), map_location=map_location, weights_only=True)
        if checkpoint.get("format") != "neon-foundation-v1":
            raise ValueError("Unsupported foundation checkpoint format")
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.calibration = checkpoint["calibration"]
        model.checkpoint_metadata = checkpoint.get("metadata", {})
        return model.to(map_location)


def save_checkpoint(path, model, calibration=None, metadata=None):
    from .policy import load_calibration
    calibration = copy.deepcopy(calibration or model.calibration or load_calibration())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "neon-foundation-v1", "model_config": copy.deepcopy(model.model_config),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "calibration": calibration, "metadata": dict(metadata or {}),
    }
    torch.save(checkpoint, destination)
    return destination


def validate_gait_import(onnx_path, samples=32, seed=0):
    """Numerical parity check, including the observation normalization and LN."""
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    model = GaitNetwork().eval()
    model.import_onnx(onnx_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    errors = []
    for _ in range(samples):
        observation = rng.normal(size=(1, 100)).astype(np.float32)
        reference = session.run(None, {"actor_obs": observation})[0]
        with torch.inference_mode():
            actual = model(torch.from_numpy(observation)).numpy()
        errors.append(float(np.max(np.abs(reference - actual))))
    return {"samples": samples, "max_abs_error": max(errors), "passed": max(errors) < 2e-5}
