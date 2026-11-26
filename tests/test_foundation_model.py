"""Independent contracts for neural transfer, deployment, and recurrent state."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foundation.model import FoundationNet, save_checkpoint, validate_gait_import
from foundation.policy import ObservationEncoder, Policy


ROOT = Path(__file__).resolve().parents[1]
GAIT = ROOT / "third_party/holosoma/fastsac_g1_29dof.onnx"


@pytest.fixture
def encoder():
    if not (ROOT / "third_party/g1/g1.xml").exists():
        pytest.skip("The public G1 calibration asset is not installed")
    return ObservationEncoder()


def public_observation(encoder, *, reversed_order=False, reset_count=0):
    names = list(encoder.joint_names)
    positions = encoder.home.copy()
    velocities = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    if reversed_order:
        names.reverse()
        positions = positions[::-1].copy()
        velocities = velocities[::-1].copy()
    return {
        "schema_version": 1, "player": 0, "time": 0., "reset_count": reset_count,
        "proprio": {"joint_names": names, "q": positions, "dq": velocities,
                    "base_quat": np.array([1., 0., 0., 0.]), "gyro": np.zeros(3),
                    "foot_contacts": [1, 1]},
        "visual": None, "tasks": [],
        "game": {"health": 100., "score": 0., "alive": True,
                 "stun_remaining": 0., "cooldown_remaining": 0., "remaining": 60.},
    }


def tensor_batch(encoded):
    return {key: torch.from_numpy(value)[None] for key, value in encoded.items()}


def test_pinned_gait_transfers_observation_normalization_and_weights():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if not GAIT.exists():
        pytest.skip("Pinned Holosoma asset is not installed")
    torch.set_num_threads(1)
    report = validate_gait_import(GAIT, samples=64, seed=719)
    assert report["passed"], report
    assert report["max_abs_error"] < 2e-5


def test_public_joint_order_is_canonical_and_output_restores_order(encoder):
    canonical = public_observation(encoder)
    reversed_observation = public_observation(encoder, reversed_order=True)
    reference = encoder.encode(canonical)
    other = ObservationEncoder(encoder.calibration).encode(reversed_observation)
    for key in reference:
        np.testing.assert_array_equal(reference[key], other[key])
    requested = np.linspace(-0.1, 0.1, 29, dtype=np.float32)
    ordered = encoder.action_to_targets(requested, canonical["proprio"]["joint_names"])
    reversed_targets = encoder.action_to_targets(requested, reversed_observation["proprio"]["joint_names"])
    np.testing.assert_array_equal(ordered, reversed_targets[::-1])
    np.testing.assert_allclose(
        encoder.targets_to_action(reversed_targets, reversed_observation["proprio"]["joint_names"]),
        requested, atol=2e-6,
    )


def test_standing_walking_transition_and_reset_preserve_gait_memory(encoder):
    encoder.encode(public_observation(encoder))
    raw_gait = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
    encoder.commit(raw_gait, [0., 0., 0.])
    np.testing.assert_allclose(encoder.phase, [np.pi, np.pi])
    assert encoder.standing
    raw_gait[:] = 99.  # Stored state must not alias the caller's output.
    np.testing.assert_allclose(encoder.last, np.linspace(-0.5, 0.5, 29), atol=1e-7)
    encoder.commit(np.ones(29), [.15, 0., 0.])
    np.testing.assert_allclose(encoder.phase, [0., np.pi])
    assert not encoder.standing
    reset = encoder.encode(public_observation(encoder, reset_count=1))
    np.testing.assert_array_equal(reset["proprio"][:29], np.zeros(29))
    np.testing.assert_allclose(encoder.phase, [0., np.pi])


def test_deployed_policy_remembers_raw_gait_instead_of_aiming_residual(encoder, monkeypatch):
    class PredictableNetwork(torch.nn.Module):
        calibration = encoder.calibration

        def forward(self, batch):
            shape = (len(batch["proprio"]), 29)
            return {"action": torch.full(shape, 1.2),
                    "gait_action": torch.full(shape, .4),
                    "command": torch.zeros((shape[0], 3)),
                    "fire_logit": torch.full((shape[0],), -4.)}

    monkeypatch.setattr(FoundationNet, "from_checkpoint", lambda *args, **kwargs: PredictableNetwork())
    policy = Policy(checkpoint="unused.pt")
    observation = public_observation(encoder)
    action = policy.act(observation)
    np.testing.assert_allclose(policy.encoder.last, .4)
    actual = policy.encoder.targets_to_action(action["joint_targets"], observation["proprio"]["joint_names"])
    assert not np.allclose(actual, policy.encoder.last)
    assert action["fire"] is False


def test_checkpoint_roundtrip_safe_metadata_and_self_contained_calibration(encoder, tmp_path, monkeypatch):
    torch.manual_seed(91)
    model = FoundationNet().eval()
    inputs = tensor_batch(encoder.encode(public_observation(encoder)))
    with torch.inference_mode():
        expected = model(inputs)
    metadata = {"torch": str(torch.__version__), "stage": "behavior_cloning",
                "arguments": {"seed": 91, "data": ["episode-a.npz"]},
                "metrics": {"validation": {"loss": 0.1}},
                "acceptance": "unverified_requires_closed_loop"}
    path = save_checkpoint(tmp_path / "network.pt", model, encoder.calibration, metadata)
    # Loading/inference must not depend on the original ONNX or XML paths.
    monkeypatch.setattr("foundation.policy.load_calibration", lambda: pytest.fail("Unexpected asset access"))
    loaded = FoundationNet.from_checkpoint(path).eval()
    assert loaded.checkpoint_metadata == metadata
    deployed = Policy(checkpoint=path)
    action = deployed.act(public_observation(encoder))
    assert np.asarray(action["joint_targets"]).shape == (29,)
    assert np.isfinite(action["joint_targets"]).all()
    with torch.inference_mode():
        actual = loaded(inputs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0., atol=0.)


def test_wrist_limits_use_radians_and_leave_other_joints_unchanged(encoder):
    model = FoundationNet().eval()
    inputs = tensor_batch(encoder.encode(public_observation(encoder)))
    # The model sees joint displacement in radians, but emits normalized actions.
    inputs["proprio"][:, 64:66] = torch.tensor([[.25, -.1]])
    with torch.no_grad():
        model.residual_head.bias[-2:] = torch.tensor([3., -3.])
        unrestricted = model(inputs)
        model.set_wrist_error_limits((.08, .05))
        limited = model(inputs)
    torch.testing.assert_close(limited["action"][:, -2:] * .25,
                               torch.tensor([[.33, -.15]]), rtol=0., atol=1e-7)
    torch.testing.assert_close(limited["action"][:, :-2], unrestricted["action"][:, :-2])
    torch.testing.assert_close(limited["wrist_candidate"], unrestricted["action"][:, -2:])
    assert model.model_config["wrist_error_limits"] == [.08, .05]


def test_saturated_wrist_candidates_keep_auxiliary_training_gradients(encoder):
    model = FoundationNet(wrist_error_limits=(.08, .05))
    inputs = tensor_batch(encoder.encode(public_observation(encoder)))
    with torch.no_grad():
        model.residual_head.bias[-2:] = torch.tensor([2., -2.])
    prediction = model(inputs)
    # Executed actions saturate, but learning a distant desired target must work.
    assert torch.all(torch.abs(prediction["wrist_candidate"]) > torch.tensor([.32, .20]))
    prediction["wrist_candidate"].square().mean().backward()
    gradient = model.residual_head.bias.grad[-2:]
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient.abs() > 1e-5)


def test_legacy_checkpoint_defaults_to_no_wrist_limit_and_new_limit_roundtrips(encoder, tmp_path):
    model = FoundationNet().eval()
    assert model.wrist_error_limits is None
    inputs = tensor_batch(encoder.encode(public_observation(encoder)))
    path = save_checkpoint(tmp_path / "legacy.pt", model, encoder.calibration)
    legacy = torch.load(path, weights_only=True)
    # Earlier checkpoints predate the optional config field and new output.
    legacy["model_config"].pop("wrist_error_limits")
    torch.save(legacy, path)
    loaded = FoundationNet.from_checkpoint(path).eval()
    assert loaded.wrist_error_limits is None
    with torch.inference_mode():
        torch.testing.assert_close(loaded(inputs)["action"], model(inputs)["action"], rtol=0., atol=0.)
    loaded.set_wrist_error_limits((.08, .05))
    bounded_path = save_checkpoint(tmp_path / "bounded.pt", loaded, encoder.calibration)
    restored = FoundationNet.from_checkpoint(bounded_path).eval()
    assert restored.wrist_error_limits == (.08, .05)
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs)["action"], loaded(inputs)["action"], rtol=0., atol=0.)
