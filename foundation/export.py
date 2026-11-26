"""Export one end-to-end ONNX graph, verify parity, and build an offline ZIP.

Example:
    python -m foundation.export --checkpoint artifacts/run/best_offline.pt \
        --output artifacts/export-v1 --data artifacts/teacher-data

Export parity is a numerical check, not a task/stability acceptance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import FoundationNet
from .onnx_policy import INPUT_NAMES, OUTPUT_NAMES


ROOT = Path(__file__).resolve().parents[1]


class ExportGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, proprio, image, tasks, task_mask, game):
        prediction = self.model(dict(zip(INPUT_NAMES, (proprio, image, tasks, task_mask, game), strict=True)))
        return tuple(prediction[name] for name in OUTPUT_NAMES)


def synthetic_inputs(rng, index=0):
    """Finite tensors exercise full, partial, and empty active task masks."""
    mask = np.zeros((1, 20), dtype=np.float32)
    mask[:, :[0, 1, 2, 7, 20][index % 5]] = 1.
    proprio = rng.normal(0., .3, (1, 100)).astype(np.float32)
    proprio[:, 32:35] = 0.  # Command slots are supplied internally by the network.
    return {"proprio": proprio,
            "image": rng.uniform(-1., 1., (1, 4, 48, 64)).astype(np.float32),
            "tasks": rng.uniform(-1., 1., (1, 20, 3, 24, 24)).astype(np.float32),
            "task_mask": mask,
            "game": rng.uniform(0., 1., (1, 8)).astype(np.float32)}


def validation_inputs(data, samples, seed=872):
    rng = np.random.default_rng(seed)
    files = sorted({file for path in data for file in
                    ([Path(path)] if Path(path).is_file() else Path(path).rglob("*.npz"))})
    if data and not files:
        raise ValueError("No episode NPZ files found for export validation")
    if not files:
        for i in range(samples):
            yield synthetic_inputs(rng, i)
        return
    chosen = rng.integers(0, len(files), size=samples)
    # Load each compressed episode once; keep no large cross-episode cache.
    for file_index in np.unique(chosen):
        with np.load(files[int(file_index)], allow_pickle=False) as datafile:
            arrays = {name: datafile[name] for name in
                      ("proprio", "game", "image_frames", "image_index", "task_frames", "task_mask_frames", "task_index")}
        rows = rng.integers(0, len(arrays["proprio"]), size=int(np.sum(chosen == file_index)))
        for row in rows:
            task_index = arrays["task_index"][row]
            yield {"proprio": arrays["proprio"][row:row+1].astype(np.float32),
                   "game": arrays["game"][row:row+1].astype(np.float32),
                   "image": arrays["image_frames"][arrays["image_index"][row]][None].astype(np.float32),
                   "tasks": arrays["task_frames"][task_index][None].astype(np.float32),
                   "task_mask": arrays["task_mask_frames"][task_index][None].astype(np.float32)}


def validate_export(model, onnx_path, *, data=(), samples=64, tolerance=5e-5):
    if samples < 1:
        raise ValueError("At least one parity sample is required")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    if {item.name for item in session.get_inputs()} != set(INPUT_NAMES):
        raise RuntimeError("Export dropped or renamed a public input")
    errors = {name: 0. for name in OUTPUT_NAMES}
    fire_decision_mismatches = 0
    inference_times = []
    for inputs in validation_inputs(data, samples):
        with torch.inference_mode():
            expected = model({name: torch.from_numpy(value) for name, value in inputs.items()})
        start = time.perf_counter()
        outputs = session.run(list(OUTPUT_NAMES), inputs)
        inference_times.append(time.perf_counter() - start)
        for name, actual in zip(OUTPUT_NAMES, outputs, strict=True):
            reference = expected[name].detach().cpu().numpy()
            if not np.isfinite(actual).all():
                raise RuntimeError(f"Nonfinite exported {name}")
            errors[name] = max(errors[name], float(np.max(np.abs(reference - actual))))
            if name == "fire_logit":
                fire_decision_mismatches += int(np.sum((reference > 0.) != (actual > 0.)))
    passed = max(errors.values()) <= tolerance and fire_decision_mismatches == 0
    return {"samples": samples, "input_source": "public_episode_observations" if data else "synthetic_tensors",
            "absolute_tolerance": tolerance, "max_abs_error": errors,
            "fire_decision_mismatches": fire_decision_mismatches,
            "onnx_inference_ms_mean": float(np.mean(inference_times) * 1000),
            "onnx_inference_ms_p95": float(np.percentile(inference_times, 95) * 1000),
            "passed": passed, "closed_loop_acceptance": "not_checked_by_export"}


def export_checkpoint(checkpoint, output, *, data=(), samples=64, name="Neon foundation network"):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve prior exports: {output}")
    if not 1 <= len(name.strip()) <= 60:
        raise ValueError("Submission name must contain 1-60 characters")
    torch.set_num_threads(1)
    model = FoundationNet.from_checkpoint(checkpoint, map_location="cpu").eval()
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    model_path = output / "models/foundation.onnx"
    inputs = synthetic_inputs(np.random.default_rng(0), 2)
    graph = ExportGraph(model).eval()
    with torch.inference_mode():
        torch.onnx.export(graph, tuple(torch.from_numpy(inputs[name]) for name in INPUT_NAMES),
                          str(model_path), input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                          opset_version=17, dynamo=False, external_data=False)
    exported = onnx.load(str(model_path))
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for key, value in (("format", "neon-foundation-onnx-v1"), ("checkpoint_sha256", checkpoint_hash),
                       ("control_hz", "50"), ("runtime", "onnxruntime-cpu")):
        entry = exported.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(exported, full_check=True)
    onnx.save_model(exported, str(model_path), save_as_external_data=False)
    report = validate_export(model, model_path, data=data, samples=samples)
    (output / "export_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"ONNX export failed parity: {report}")
    calibration = model.calibration
    metadata = {"format": "neon-foundation-onnx-v1", "checkpoint_sha256": checkpoint_hash,
                "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "model_config": model.model_config, "calibration": calibration,
                "source_metadata": model.checkpoint_metadata, "export_validation": report,
                "input_names": list(INPUT_NAMES), "output_names": list(OUTPUT_NAMES),
                "batch_size": 1, "control_hz": 50,
                "acceptance": model.checkpoint_metadata.get("acceptance", "unverified_requires_closed_loop")}
    (output / "models/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"name": name, "api_version": 1, "entry_point": "policy:Policy",
                "environment": "arena-python-v1", "robot_id": "unitree_g1_29dof",
                "control_mode": "joint_position", "joint_unit": "rad",
                "description": "Single public-observation neural graph; see export validation and training provenance."}
    (output / "submission.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    robot = {"id": "unitree_g1_29dof", "api_version": 1, "dof": 29,
             "control_mode": "joint_position", "joint_unit": "rad",
             "joint_names": calibration["dof_names"], "home": calibration["home"],
             "kp": calibration["kp"], "kd": calibration["kd"],
             "control_hz": 50, "camera_hz": 10, "base_quaternion_order": "wxyz"}
    (output / "robot.json").write_text(json.dumps(robot, indent=2), encoding="utf-8")
    (output / "policy.py").write_text('"""Offline submission entry point."""\nfrom foundation.onnx_policy import Policy\n', encoding="utf-8")
    (output / "foundation").mkdir()
    for file in ("__init__.py", "policy.py", "onnx_policy.py"):
        (output / "foundation" / file).write_bytes((ROOT / "foundation" / file).read_bytes())
    license_source = ROOT / "third_party/holosoma/LICENSE"
    if not license_source.exists():
        raise FileNotFoundError("Holosoma license is required for redistribution")
    (output / "LICENSES").mkdir()
    (output / "LICENSES/Holosoma-Apache-2.0.txt").write_bytes(license_source.read_bytes())
    readme = (
        "# Neon foundation neural policy\n\n"
        "CPU inference requires NumPy, Pillow, and ONNX Runtime, supplied by arena-python-v1.\n"
        "The single models/foundation.onnx graph includes visual/task encoders and the G1 gait network.\n"
        "The adapter handles public observation tensors, recurrent gait memory, joint limits, and boolean fire decoding.\n"
        "No teacher, target detector, inverse kinematics, simulator, PyTorch, or original ONNX checkpoint is loaded at inference.\n\n"
        "Run at 50 Hz. reset_count changes clear recurrent state and cached observations.\n"
        "Numerical export parity does not establish standing, scoring, or city navigation success.\n"
        "Training provenance and checkpoint hash are recorded in models/metadata.json; parity in export_validation.json.\n"
        "Holosoma-derived weights retain the bundled Apache-2.0 license.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(file for file in output.rglob("*") if file.is_file())
    package = output / "submission.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(output).as_posix())
    return {"output": str(output), "model": str(model_path), "submission": str(package),
            "checkpoint_sha256": checkpoint_hash, "onnx_sha256": metadata["onnx_sha256"],
            "submission_sha256": hashlib.sha256(package.read_bytes()).hexdigest(), "validation": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--name", default="Neon foundation network")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output, data=args.data,
                                      samples=args.samples, name=args.name), indent=2), flush=True)


if __name__ == "__main__":
    main()
