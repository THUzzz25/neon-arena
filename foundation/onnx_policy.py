"""Offline CPU deployment of the single foundation graph; no Torch dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .policy import ObservationEncoder


ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = ("proprio", "image", "tasks", "task_mask", "game")
OUTPUT_NAMES = ("action", "command", "fire_logit", "gait_action")


class Policy:
    def __init__(self, player_id=0, config=None, model_path=None, metadata_path=None):
        self.player_id = player_id
        model_path = Path(model_path or ROOT / "models/foundation.onnx")
        metadata_path = Path(metadata_path or model_path.with_name("metadata.json"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != "neon-foundation-onnx-v1":
            raise ValueError("Unsupported foundation ONNX metadata")
        model_bytes = model_path.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != self.metadata["onnx_sha256"]:
            raise ValueError("Foundation ONNX checksum does not match metadata")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
        if {value.name for value in self.session.get_inputs()} != set(INPUT_NAMES):
            raise ValueError("Unexpected foundation ONNX inputs")
        if {value.name for value in self.session.get_outputs()} != set(OUTPUT_NAMES):
            raise ValueError("Unexpected foundation ONNX outputs")
        self.encoder = ObservationEncoder(self.metadata["calibration"])
        self.last_prediction = None

    def act(self, observation):
        inputs = {name: np.ascontiguousarray(value[None], dtype=np.float32)
                  for name, value in self.encoder.encode(observation).items()}
        outputs = self.session.run(list(OUTPUT_NAMES), inputs)
        self.last_prediction = {name: value[0].copy() for name, value in zip(OUTPUT_NAMES, outputs, strict=True)}
        if not all(np.isfinite(value).all() for value in self.last_prediction.values()):
            raise ValueError("Nonfinite foundation network prediction")
        targets = self.encoder.action_to_targets(self.last_prediction["action"], observation["proprio"]["joint_names"])
        self.encoder.commit(self.last_prediction["gait_action"], self.last_prediction["command"])
        return {"joint_targets": targets, "fire": bool(self.last_prediction["fire_logit"] > 0.)}
