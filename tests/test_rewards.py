import importlib

import pytest


@pytest.fixture(params=["embodied_arena", "participant.embodied_arena"], ids=["main", "participant"])
def arena(request):
    config_module = importlib.import_module(f"{request.param}.config")
    engine_module = importlib.import_module(f"{request.param}.engine")
    config = config_module.MatchConfig(
        scene="room", duration=.06, shot_cooldown=.02,
        private_points=13.25, public_points=7.5,
        hit_damage=100., respawn_delay=.04,
    )
    environment = engine_module.Arena(config, cameras=False)
    yield environment
    environment.close()


def actions(arena, *firing_players):
    return [{"joint_targets": robot.home, "fire": pid in firing_players}
            for pid, robot in enumerate(arena.robots)]


def set_hits(monkeypatch, arena, hits):
    rules = importlib.import_module(f"{type(arena).__module__.rsplit('.', 1)[0]}.rules")
    monkeypatch.setattr(arena, "cast_shot", lambda pid: rules.Hit(*hits[pid]))


def test_private_reward_is_per_step_and_terminal_reward_is_preserved(arena, monkeypatch):
    set_hits(monkeypatch, arena, {0: ("target", "blue_cup"), 1: ("target", "green_cup")})

    first, done = arena.step(actions(arena, 0))
    assert first == [13.25, 0.0]
    assert done is False
    assert arena.observation(0)["game"]["score"] == 13.25

    idle, done = arena.step(actions(arena))
    assert idle == [0.0, 0.0]
    assert done is False
    assert first == [13.25, 0.0]

    final, done = arena.step(actions(arena, 1))
    assert final == [0.0, 13.25]
    assert done is True
    assert [first[pid] + idle[pid] + final[pid] for pid in (0, 1)] == [
        player.score for player in arena.referee.players
    ]


def test_shared_reward_keeps_fractional_points_and_is_not_repeated(arena, monkeypatch):
    set_hits(monkeypatch, arena, {0: ("target", "red_cup"), 1: ("target", "red_cup")})

    rewards, done = arena.step(actions(arena, 0, 1))
    assert rewards == [3.75, 3.75]
    assert done is False

    repeated, done = arena.step(actions(arena, 0, 1))
    assert repeated == [0.0, 0.0]
    assert [player.score for player in arena.referee.players] == [3.75, 3.75]
    assert [player.health for player in arena.referee.players] == [95.0, 95.0]


def test_damage_death_and_respawn_do_not_shape_rewards(arena, monkeypatch):
    set_hits(monkeypatch, arena, {0: ("player", 1), 1: ("player", 0)})

    rewards, done = arena.step(actions(arena, 0, 1))
    assert rewards == [0.0, 0.0]
    assert done is False
    assert [player.health for player in arena.referee.players] == [0.0, 0.0]

    rewards, done = arena.step(actions(arena))
    assert rewards == [0.0, 0.0]
    assert done is False
    assert arena.reset_counts == [1, 1]
    assert [player.health for player in arena.referee.players] == [100.0, 100.0]


def test_invalid_action_returns_zero_reward(arena):
    invalid_actions = actions(arena)
    invalid_actions[0]["joint_targets"] = [float("nan")] * 29

    rewards, done = arena.step(invalid_actions)

    assert rewards == [0.0, 0.0]
    assert done is False
    assert any(event["type"] == "invalid_action" for event in arena.referee.events)
