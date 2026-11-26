"""Export parity and isolated NumPy/Pillow/ORT-only submission verification."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from foundation.export import export_checkpoint
from foundation.model import FoundationNet, save_checkpoint
from foundation.onnx_policy import Policy as OnnxPolicy
from foundation.policy import ObservationEncoder, Policy as TorchPolicy


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    gait = ROOT / "third_party/holosoma/fastsac_g1_29dof.onnx"
    if not gait.exists():
        pytest.skip("Pinned Holosoma assets are not installed")
    torch.set_num_threads(1)
    torch.manual_seed(456)
    model = FoundationNet(gait).eval()
    # Exercise visual/task branches: zero heads could hide incorrect export paths.
    for head in (model.command_head, model.residual_head, model.fire_head):
        torch.nn.init.normal_(head.weight, std=.003)
    root = tmp_path_factory.mktemp("foundation-export")
    checkpoint = save_checkpoint(root / "candidate.pt", model, metadata={
        "torch": str(torch.__version__), "stage": "test_candidate", "acceptance": "not_evaluated"})
    result = export_checkpoint(checkpoint, root / "bundle", samples=10)
    return root, checkpoint, result


def observation(calibration, time=0., reset_count=0):
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[30:45, 70:85, 0] = 220
    return {"schema_version": 1, "time": time, "player": 0, "reset_count": reset_count,
            "proprio": {"joint_names": calibration["dof_names"], "q": np.asarray(calibration["home"]),
                        "dq": np.zeros(29), "base_quat": np.array([1., 0., 0., 0.]),
                        "gyro": np.zeros(3), "foot_contacts": [1, 1]},
            "visual": {"time": int(round(time * 50)) // 5 * .1, "rgb": rgb,
                       "depth": np.full((120, 160), 3., dtype=np.float32), "fovy_degrees": 70.},
            "tasks": [{"id": "sample", "kind": "public", "text": "Match reference",
                       "reference_rgb": np.full((96, 96, 3), [220, 10, 10], dtype=np.uint8)}],
            "game": {"health": 100., "score": 0., "alive": True, "stun_remaining": 0.,
                     "cooldown_remaining": 0., "remaining": 60. - time}}


def test_single_graph_parity_and_recurrent_deployment_actions(exported):
    root, checkpoint, result = exported
    assert result["validation"]["passed"]
    assert result["validation"]["fire_decision_mismatches"] == 0
    with zipfile.ZipFile(result["submission"]) as archive:
        assert [name for name in archive.namelist() if name.endswith(".onnx")] == ["models/foundation.onnx"]
        assert "foundation/model.py" not in archive.namelist()
        assert "LICENSES/Holosoma-Apache-2.0.txt" in archive.namelist()
    onnx_policy = OnnxPolicy(model_path=result["model"])
    torch_policy = TorchPolicy(checkpoint=checkpoint)
    for frame in range(25):
        obs = observation(onnx_policy.encoder.calibration, frame * .02, reset_count=int(frame >= 17))
        ort_action = onnx_policy.act(obs)
        torch_action = torch_policy.act(obs)
        np.testing.assert_allclose(ort_action["joint_targets"], torch_action["joint_targets"], atol=2e-5, rtol=0.)
        assert ort_action["fire"] == torch_action["fire"]
        np.testing.assert_allclose(onnx_policy.encoder.last, torch_policy.encoder.last, atol=5e-5, rtol=0.)


def test_submission_import_and_inference_with_training_dependencies_blocked(exported, tmp_path):
    root, checkpoint, result = exported
    with zipfile.ZipFile(result["submission"]) as archive:
        archive.extractall(tmp_path)
    code = r'''
import importlib.abc, json, sys
from pathlib import Path
class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco', 'onnx', 'stable_baselines3'}:
            raise ImportError('Forbidden inference dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingDependencies())
sys.path.insert(0, sys.argv[1])
from policy import Policy
policy = Policy(player_id=0)
c = policy.encoder.calibration
observation = {'time': 0., 'player': 0, 'reset_count': 0, 'visual': None, 'tasks': [],
 'proprio': {'joint_names': c['dof_names'], 'q': c['home'], 'dq': [0.] * 29,
  'base_quat': [1., 0., 0., 0.], 'gyro': [0., 0., 0.], 'foot_contacts': [1, 1]},
 'game': {'health': 100., 'score': 0., 'alive': True, 'stun_remaining': 0.,
  'cooldown_remaining': 0., 'remaining': 60.}}
action = policy.act(observation)
assert len(action['joint_targets']) == 29
assert type(action['fire']) is bool
assert not {'torch', 'mujoco', 'onnx', 'stable_baselines3'}.intersection(sys.modules)
print(json.dumps({'joint_count': len(action['joint_targets']), 'fire': action['fire'], 'torch_imported': 'torch' in sys.modules}))
'''
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)],
                               text=True, capture_output=True, timeout=45, check=True)
    assert json.loads(completed.stdout)["torch_imported"] is False


def test_model_checksum_rejects_mixed_export_files(exported, tmp_path):
    root, checkpoint, result = exported
    original = Path(result["model"])
    model_path = tmp_path / "foundation.onnx"
    model_path.write_bytes(original.read_bytes() + b"different artifact")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_bytes(original.with_name("metadata.json").read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        OnnxPolicy(model_path=model_path, metadata_path=metadata_path)


def test_export_does_not_overwrite_a_version(exported):
    root, checkpoint, result = exported
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, root / "bundle", samples=1)
