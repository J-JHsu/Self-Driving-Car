from collections import defaultdict

import pygame
import pytest

from car_hum import HumanGame, read_keys
from car_sim import config, ui
from car_sim.car import Car
from car_sim.world import World


@pytest.fixture(scope="module", autouse=True)
def pygame_ready():
    pygame.init()
    yield
    pygame.quit()


def make_surface():
    return pygame.Surface((config.GAME_W, config.GAME_H + config.PANEL_H))


def test_ui_draws_without_error():
    surface = make_surface()
    font = pygame.font.Font(None, 24)
    ui.draw_panel(surface, font, ["Score: 0", "Max: 0"])
    for steer in (0, 0.5, 1, -1):
        ui.draw_steering_wheel(surface, steer)
    for value in (0, 0.5, 1):
        ui.draw_pedal(surface, ui.THROTTLE_RECT, value, config.THROTTLE_COLOR, "Throttle", font)
        ui.draw_pedal(surface, ui.BRAKE_RECT, value, config.BRAKE_COLOR, "Brake", font)


def test_draw_sensors_draws_rays():
    surface = make_surface()
    surface.fill((0, 0, 0))
    car = Car()
    car.x, car.y = 400.0, 300.0
    ui.draw_sensors(surface, World(), car)
    assert surface.get_at((int(car.x) + 40, int(car.y)))[:3] == config.SENSOR_COLOR


def test_ghost_car_is_drawn_fainter():
    solid, ghost = make_surface(), make_surface()
    solid.fill((0, 0, 0))
    ghost.fill((0, 0, 0))
    car = Car()
    car.x, car.y = 400.0, 300.0
    car.draw(solid, config.AI_CAR_COLOR)
    car.draw(ghost, config.AI_CAR_COLOR, alpha=config.GHOST_CAR_ALPHA)
    full = solid.get_at((400, 300))[:3]
    faint = ghost.get_at((400, 300))[:3]
    assert full == config.AI_CAR_COLOR
    assert 0 < sum(faint) < sum(full)


def test_ai_window_draws_many_cars_and_lists_them():
    from car_sim.vec_env import VecDrivingEnv
    from self_driving_car_ai import _draw

    screen = pygame.display.set_mode((config.GAME_W, config.GAME_H + config.PANEL_H))
    font = pygame.font.Font(None, 24)
    vec = VecDrivingEnv(World(), 5, use_route=True)
    vec.reset()
    for _ in range(30):
        vec.step([0, 1, 2, 3, 4])
    texts = []
    real_panel = ui.draw_panel
    ui.draw_panel = lambda s, f, lines: (texts.extend(lines), real_panel(s, f, lines))
    try:
        _draw(screen, font, vec, True, True, None, False, 3, 100.0, 50.0, 0.5, [True, False],
              False, False)
    finally:
        ui.draw_panel = real_panel
    assert any("Cars: 5" in t for t in texts)
    limit = ui.WHEEL_CENTER[0] - ui.WHEEL_RADIUS - 10
    assert all(font.size(t)[0] + 15 < limit for t in texts)
    assert len(texts) <= ui.MAX_PANEL_LINES


def test_read_keys():
    keys = defaultdict(bool)
    assert read_keys(keys) == (0.0, 0.0, 0.0)
    keys[pygame.K_UP] = True
    assert read_keys(keys) == (1.0, 0.0, 0.0)
    keys.clear()
    keys[pygame.K_s] = True
    keys[pygame.K_LEFT] = True
    assert read_keys(keys) == (0.0, 1.0, -1.0)
    keys.clear()
    keys[pygame.K_a] = True
    keys[pygame.K_d] = True
    assert read_keys(keys)[2] == 0.0
    keys.clear()
    keys[pygame.K_RIGHT] = True
    assert read_keys(keys)[2] == 1.0


def test_score_only_increases_when_moving():
    game = HumanGame()
    for _ in range(30):
        game.update(1 / 60, 0.0, 0.0, 0.0)
    assert game.score == 0
    for _ in range(60):
        game.update(1 / 60, 1.0, 0.0, 0.0)
    assert game.score > 0
    assert game.max_score == game.score


def test_crash_sets_game_over_and_freezes():
    game = HumanGame()
    game.world.add_segment((config.START_X + 10, 0), (config.START_X + 10, config.GAME_H))
    game.update(1 / 60, 1.0, 0.0, 0.0)
    assert game.game_over
    x, score = game.car.x, game.score
    game.update(1 / 60, 1.0, 0.0, 0.0)
    assert (game.car.x, game.score) == (x, score)


def test_restart_resets_score_but_keeps_max():
    game = HumanGame()
    for _ in range(60):
        game.update(1 / 60, 1.0, 0.0, 0.0)
    best = game.score
    game.world.add_segment((int(game.car.x) + 20, 0), (int(game.car.x) + 20, config.GAME_H))
    game.update(1 / 60, 1.0, 0.0, 0.0)
    assert game.game_over
    game.world.clear()
    game.reset()
    assert not game.game_over
    assert game.score == 0
    assert game.max_score >= best


def key_event(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def click_event(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def test_markers_draw_at_points():
    surface = make_surface()
    surface.fill(config.BG_COLOR)
    world = World()
    world.set_start((300, 200))
    world.set_goal((900, 500))
    ui.draw_markers(surface, world, None)
    assert surface.get_at((300, 200))[:3] == config.START_MARKER_COLOR
    assert surface.get_at((900, 500))[:3] == config.GOAL_MARKER_COLOR
    ui.draw_markers(surface, world, "goal")  # highlight while placing runs too


def test_draw_route_runs():
    surface = make_surface()
    ui.draw_route(surface, [(100, 400), (500, 400), (1300, 400)])
    ui.draw_route(surface, [])
    ui.draw_route(surface, [(100, 400)])
    assert surface.get_at((300, 400))[:3] == config.ROUTE_COLOR


def test_placement_keys_and_click():
    world = World()
    mode = ui.handle_placement_event(key_event(pygame.K_1), world, None)
    assert mode == "start"
    mode = ui.handle_placement_event(click_event((300, 300)), world, mode)
    assert mode is None and world.start == (300, 300)
    mode = ui.handle_placement_event(key_event(pygame.K_2), world, None)
    assert mode == "goal"
    mode = ui.handle_placement_event(click_event((900, 500)), world, mode)
    assert mode is None and world.goal == (900, 500)


def test_click_without_mode_does_nothing():
    world = World()
    assert ui.handle_placement_event(click_event((300, 300)), world, None) is None
    assert world.start == (config.START_X, config.START_Y)


def test_placement_right_click_ignored():
    world = World()
    mode = ui.handle_placement_event(click_event((300, 300), button=3), world, "start")
    assert mode == "start" and world.start == (config.START_X, config.START_Y)


def test_placement_escape_cancels():
    world = World()
    assert ui.handle_placement_event(key_event(pygame.K_ESCAPE), world, "start") is None


def test_placement_click_in_panel_ignored_mode_stays():
    world = World()
    mode = ui.handle_placement_event(click_event((300, config.GAME_H + 20)), world, "goal")
    assert mode == "goal"
    assert world.goal == (config.GOAL_X, config.GOAL_Y)


def test_human_wins_at_finish():
    game = HumanGame()
    game.world.set_goal((config.START_X + 25, config.START_Y))
    game.update(1 / 60, 0.0, 0.0, 0.0)
    assert game.won and game.game_over


def test_human_win_freezes_and_reset_restarts_from_start():
    game = HumanGame()
    game.world.set_goal((config.START_X + 25, config.START_Y))
    game.update(1 / 60, 0.0, 0.0, 0.0)
    elapsed = game.elapsed
    game.update(1 / 60, 1.0, 0.0, 0.0)
    assert game.elapsed == elapsed
    game.world.set_start((300, 300))
    game.reset()
    assert (game.car.x, game.car.y) == (300, 300)
    assert not game.won and not game.game_over and game.elapsed == 0.0


def test_human_spawns_at_world_start():
    game = HumanGame()
    game.world.set_start((300, 300))
    game.reset()
    assert (game.car.x, game.car.y) == (300, 300)


def test_panel_text_never_reaches_the_controls():
    font = pygame.font.Font(None, 24)
    limit = ui.WHEEL_CENTER[0] - ui.WHEEL_RADIUS - 10
    worst_case_lines = list(ui.ALL_HINTS) + [
        "Score: 999999   Max score: 999999   Time: 9999.9s",
        "Episode: 99999   Time alive: 3000.0s   Longest: 3000.0s   Avg time (last 20): 3000.0s",
        "Reached finish: 20 of last 20 episodes   Epsilon: 0.050   Speed: normal   Cars: 64",
        "Click the map to place the FINISH  (Esc cancels)",
        "No route from start to finish: move a point or open a gap",
    ]
    for line in worst_case_lines:
        assert font.size(line)[0] + 15 < limit, line


def test_panel_is_tall_enough_for_its_lines_and_controls_fit():
    assert ui.MAX_PANEL_LINES * 22 + 10 <= config.PANEL_H
    assert ui.WHEEL_CENTER[1] + ui.WHEEL_RADIUS <= ui.PANEL_TOP + config.PANEL_H
    assert ui.BRAKE_RECT[0] + ui.BRAKE_RECT[2] < ui.MESSAGE_POS[0]
    assert ui.MESSAGE_POS[1] + 40 <= ui.PANEL_TOP + config.PANEL_H
