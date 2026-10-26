import os
import threading
import time

import pygame

from self_driving_car_ai import train


def test_short_headless_training_runs(tmp_path):
    path = str(tmp_path / "model.pt")
    result = train(headless=True, max_steps=2000, model_path=path)
    assert result["steps"] == 2000
    assert result["episodes"] >= 1
    assert os.path.exists(path)


def test_resume_from_saved_model(tmp_path):
    path = str(tmp_path / "model.pt")
    first = train(headless=True, max_steps=1500, model_path=path)
    second = train(headless=True, max_steps=500, model_path=path)
    assert second["steps"] == 500
    assert second["episodes"] >= first["episodes"] >= 1
    assert second["max_score"] >= first["max_score"]


def test_corrupt_model_starts_fresh(tmp_path, capsys):
    path = tmp_path / "model.pt"
    path.write_bytes(b"garbage, not a model")
    result = train(headless=True, max_steps=200, model_path=str(path))
    assert result["steps"] == 200
    assert capsys.readouterr().out.count("starting fresh") == 1


def test_window_close_saves_model(tmp_path):
    path = str(tmp_path / "model.pt")

    def close_window_soon():
        for _ in range(100):
            time.sleep(0.1)
            try:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            except pygame.error:  # pygame is not initialised yet
                continue

    threading.Thread(target=close_window_soon, daemon=True).start()
    result = train(headless=False, max_steps=3000, model_path=path)
    assert result["steps"] < 3000  # stopped by the close event, not by max_steps
    assert os.path.exists(path)


def test_result_has_goals_count(tmp_path):
    result = train(headless=True, max_steps=500, model_path=str(tmp_path / "m.pt"))
    assert 0 <= result["goals"] <= result["episodes"]
    assert result["no_route"] is False


def test_headless_no_route_returns_instead_of_hanging(tmp_path):
    from car_sim import config
    from car_sim.world import World

    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))
    result = train(headless=True, max_steps=500, model_path=str(tmp_path / "m.pt"), world=world)
    assert result["no_route"] is True
    assert result["steps"] == 0


def test_old_input_size_model_starts_fresh(tmp_path, monkeypatch, capsys):
    from car_sim import config
    from car_sim.dqn import DoubleDQNAgent

    path = str(tmp_path / "model.pt")
    with monkeypatch.context() as m:
        m.setattr(config, "STATE_DIM", 6)
        DoubleDQNAgent().save(path, {"epsilon": 0.5, "episode": 3, "max_score": 10.0})
    result = train(headless=True, max_steps=200, model_path=path)
    assert result["steps"] == 200
    assert capsys.readouterr().out.count("starting fresh") == 1


def test_goal_reached_counts(tmp_path):
    from car_sim import config
    from car_sim.world import World

    world = World()
    world.set_goal((config.START_X + 25, config.START_Y))
    result = train(headless=True, max_steps=300, model_path=str(tmp_path / "m.pt"), world=world)
    assert result["goals"] >= 1


def test_windowed_training_pauses_without_route_and_resumes_when_fixed(tmp_path):
    from car_sim import config
    from car_sim.world import World

    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))  # the finish is walled off
    frames = {"n": 0}
    real_get = pygame.event.get

    def scripted(*args, **kwargs):
        frames["n"] += 1
        if frames["n"] == 40:  # after some paused frames, the user presses C
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c)]
        return real_get(*args, **kwargs)

    pygame.event.get = scripted
    try:
        result = train(headless=False, max_steps=30, model_path=str(tmp_path / "m.pt"), world=world)
    finally:
        pygame.event.get = real_get
    assert result["steps"] == 30 and result["no_route"] is False
    assert frames["n"] >= 65  # about 40 paused frames passed before the 30 steps happened


def test_no_route_mode_trains_even_when_walls_block_the_finish(tmp_path):
    from car_sim import config
    from car_sim.world import World

    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))
    result = train(headless=True, max_steps=300, model_path=str(tmp_path / "m.pt"),
                   world=world, use_route=False)
    assert result["steps"] == 300 and result["no_route"] is False


def test_default_model_paths_differ_by_mode():
    from car_sim import config

    assert config.MODEL_PATH != config.MODEL_PATH_NO_ROUTE


def test_space_pauses_and_resumes_training(tmp_path):
    frames = {"n": 0}
    real_get = pygame.event.get

    def scripted(*args, **kwargs):
        frames["n"] += 1
        if frames["n"] in (5, 60):  # pause, then resume
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)]
        return real_get(*args, **kwargs)

    pygame.event.get = scripted
    try:
        result = train(headless=False, max_steps=20, model_path=str(tmp_path / "m.pt"))
    finally:
        pygame.event.get = real_get
    assert result["steps"] == 20
    assert frames["n"] >= 70  # about 55 paused frames passed while no steps happened
