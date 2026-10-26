import math

import numpy as np
import pytest

from car_sim import config
from car_sim.driving_env import DrivingEnv, compute_reward, proximity
from car_sim.world import World


def test_state_shape_and_range():
    state = DrivingEnv().reset()
    assert state.shape == (config.STATE_DIM,)
    assert state.dtype == np.float32
    assert ((state >= -1) & (state <= 1)).all()
    assert (state[:6] >= 0).all()


def test_state_has_route_values():
    env = DrivingEnv(random_spawn=False)
    env.reset()
    env.car.angle = 0.0  # facing the finish on an open map
    state = env.get_state()
    assert -1 <= state[6] <= 1
    assert 0 <= state[7] <= 1
    assert abs(state[6]) < 0.1


def test_reward_crash_is_negative():
    assert compute_reward(True, 0.5, 0.5, 0.0, 3.0, False) < 0


def test_reward_progress_positive_and_regress_negative():
    ahead = compute_reward(False, 0.5, 1.0, 0.0, 3.0, False)
    back = compute_reward(False, 0.5, 1.0, 0.0, -3.0, False)
    assert ahead > 0 > back


def test_reward_goal_is_large_and_positive():
    assert compute_reward(False, 0.5, 1.0, 0.0, 0.0, True) == config.R_GOAL


def test_reward_near_obstacle_lower_than_clear():
    near = compute_reward(False, 0.5, 0.1, 0.0, 3.0, False)
    clear = compute_reward(False, 0.5, 1.0, 0.0, 3.0, False)
    assert near < clear


def test_reward_fast_and_close_is_worse_than_slow_and_close():
    fast = compute_reward(False, 0.9, 0.1, 0.0, 3.0, False)
    slow = compute_reward(False, 0.3, 0.1, 0.0, 3.0, False)
    assert fast < slow


def test_proximity_considers_side_sensors():
    clear = np.array([1, 1, 1, 1, 1, 0.5, 0, 1], dtype=np.float32)
    left_wall = np.array([0.1, 1, 1, 1, 1, 0.5, 0, 1], dtype=np.float32)
    front_wall = np.array([1, 1, 0.2, 1, 1, 0.5, 0, 1], dtype=np.float32)
    assert proximity(clear) == pytest.approx(1.0)
    assert proximity(left_wall) < 0.5  # a close side wall counts, with a margin
    assert proximity(left_wall) == pytest.approx(0.1 + config.PROXIMITY_SIDE_MARGIN)
    assert proximity(front_wall) == pytest.approx(0.2)


def test_random_spawn_varies_and_is_free():
    world = World()
    world.add_segment((400, 100), (400, 500))
    env = DrivingEnv(world, random_spawn=True)
    spots = set()
    for _ in range(40):
        env.reset()
        assert not world.collided(env.car)
        spots.add((round(env.car.x), round(env.car.y), round(env.car.angle, 2)))
    assert len(spots) > 10


def test_spawn_off_uses_start_position():
    env = DrivingEnv(random_spawn=False)
    env.reset()
    assert (env.car.x, env.car.y) == (config.START_X, config.START_Y)
    want = env.route.direction(config.START_X, config.START_Y)
    assert abs(env.car.angle - want) <= config.START_HEADING_JITTER + 1e-9


def test_random_spawn_falls_back_when_everything_is_blocked():
    world = World()
    for y in range(0, config.GAME_H, 20):  # a dense grid of walls: no free spot
        world.add_segment((0, y), (config.GAME_W, y))
    env = DrivingEnv(world, random_spawn=True)
    state = env.reset()
    assert state.shape == (config.STATE_DIM,)
    assert (env.car.x, env.car.y) == (config.START_X, config.START_Y)


def test_step_returns_triplet():
    env = DrivingEnv(random_spawn=False)
    env.reset()
    state, reward, done = env.step(2)
    assert state.shape == (config.STATE_DIM,)
    assert isinstance(reward, float)
    assert done is False


def test_crash_ends_episode_negative():
    world = World()
    world.add_segment((config.START_X + 10, 0), (config.START_X + 10, config.GAME_H))
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    _, reward, done = env.step(2)
    assert done is True
    assert reward == config.R_CRASH
    assert env.outcome == "crash"


def test_stall_ends_episode():
    world = World()
    world.add_segment((config.START_X + 40, 0), (config.START_X + 40, config.GAME_H))
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    done, reward, steps = False, 0.0, 0
    while not done and steps < config.STALL_LIMIT + 10:
        _, reward, done = env.step(2)
        steps += 1
    assert done is True
    assert steps == config.STALL_LIMIT
    assert reward <= config.R_STALL
    assert env.outcome == "stall"


def test_reset_with_obstacle_on_spawn():
    world = World()
    world.add_segment((config.START_X - 5, 0), (config.START_X - 5, config.GAME_H))
    env = DrivingEnv(world, random_spawn=False)
    state = env.reset()
    assert state.shape == (config.STATE_DIM,)
    _, _, done = env.step(2)
    assert done is True
    env.reset()  # resetting again still works


def test_steer_map_used():
    left, right = DrivingEnv(random_spawn=False), DrivingEnv(random_spawn=False)
    left.reset()
    right.reset()
    left.car.angle = right.car.angle = 0.0  # remove the start-heading jitter
    for _ in range(10):
        left.step(0)
        right.step(4)
    assert left.car.angle < 0 < right.car.angle


def test_step_casts_each_sensor_once():
    world = World()
    calls = []
    original = world.cast_ray
    world.cast_ray = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    calls.clear()
    env.step(2)
    assert len(calls) == len(config.SENSOR_ANGLES)


def test_last_controls_recorded():
    env = DrivingEnv(random_spawn=False)
    env.reset()
    env.step(4)
    throttle, brake, steer = env.last_controls
    assert 0 <= throttle <= 1 and 0 <= brake <= 1 and 0 < steer <= 1


def test_progress_sign():
    forward, backward = DrivingEnv(random_spawn=False), DrivingEnv(random_spawn=False)
    forward.reset()
    backward.reset()
    forward.car.angle = 0.0
    backward.car.angle = math.pi
    for env in (forward, backward):
        env.car.speed = 60.0  # already rolling so one step moves visibly
    _, r_fwd, _ = forward.step(2)
    _, r_back, _ = backward.step(2)
    assert r_fwd > r_back


def test_reaching_goal_ends_episode():
    world = World()
    world.set_goal((config.START_X + 25, config.START_Y))
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    _, reward, done = env.step(2)
    assert done is True
    assert env.outcome == "goal"
    assert reward >= config.R_GOAL


def test_start_equals_goal_ends_immediately():
    world = World()
    world.set_goal((config.START_X, config.START_Y))
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    _, reward, done = env.step(2)
    assert done is True and env.outcome == "goal" and reward >= config.R_GOAL


def test_timeout_ends_episode(monkeypatch):
    monkeypatch.setattr(config, "MAX_EPISODE_STEPS", 5)
    env = DrivingEnv(random_spawn=False)
    env.reset()
    for _ in range(4):
        _, _, done = env.step(2)
        assert done is False
    _, reward, done = env.step(2)
    assert done is True
    assert env.outcome == "timeout"
    assert reward == config.R_TIMEOUT
    assert env.steps == 5


def test_reset_clears_outcome_and_steps():
    world = World()
    world.set_goal((config.START_X, config.START_Y))
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    env.step(2)
    env.reset()
    assert env.outcome == "" and env.steps == 0


def test_reset_heading_within_jitter():
    env = DrivingEnv(random_spawn=False)
    for _ in range(50):
        env.reset()
        want = env.route.direction(env.car.x, env.car.y)
        assert abs(env.car.angle - want) <= config.START_HEADING_JITTER + 1e-9


def test_reset_places_car_at_world_start():
    world = World()
    world.set_start((300, 200))
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    assert (env.car.x, env.car.y) == (300, 200)


def test_route_ok_false_when_finish_blocked_and_step_still_works():
    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))
    env = DrivingEnv(world, random_spawn=False)
    state = env.reset()
    assert env.route_ok is False
    assert state[6] == 0.0 and state[7] == 1.0
    new_state, reward, done = env.step(2)
    assert new_state.shape == (config.STATE_DIM,)
    assert isinstance(reward, float)
    assert new_state[7] == 1.0


def test_wall_drawn_mid_episode_does_not_fake_progress():
    env = DrivingEnv(random_spawn=False)
    env.reset()
    env.step(2)
    env.world.add_segment((600, 250), (600, 550))  # lengthens the route suddenly
    _, reward, _ = env.step(2)
    assert abs(reward) < 5  # no huge positive/negative jump from the rebuilt field


def test_random_spawn_is_free_and_routable():
    world = World()
    world.add_segment((700, 0), (700, 600))
    env = DrivingEnv(world, random_spawn=True)
    for _ in range(30):
        env.reset()
        assert not world.collided(env.car)
        assert env.route.reachable(env.car.x, env.car.y)


def test_start_inside_wall_does_not_raise():
    world = World()
    world.add_segment((config.START_X - 30, config.START_Y), (config.START_X + 30, config.START_Y))
    env = DrivingEnv(world, random_spawn=False)
    env.reset()
    _, _, done = env.step(2)
    assert done is True and env.outcome == "crash"


def test_no_route_mode_uses_straight_line_and_never_reports_blocked():
    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))
    env = DrivingEnv(world, random_spawn=False, use_route=False)
    state = env.reset()
    assert env.route_ok is True
    assert abs(state[6]) < 0.6  # start heading is within the jitter of the straight line
    assert state[7] == pytest.approx(1200 / config.ROUTE_DIST_SCALE, abs=0.01)
    _, reward, _ = env.step(2)
    assert isinstance(reward, float)


def test_no_route_mode_rewards_straight_line_progress():
    env = DrivingEnv(random_spawn=False, use_route=False)
    env.reset()
    env.car.angle = 0.0
    env.car.speed = 60.0
    _, reward, _ = env.step(2)
    assert reward > 0
