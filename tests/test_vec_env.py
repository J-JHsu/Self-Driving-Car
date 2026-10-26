import random

import numpy as np

from car_sim import config
from car_sim.driving_env import DrivingEnv
from car_sim.route import RouteField
from car_sim.vec_env import VecDrivingEnv
from car_sim.world import World


def crash_world():
    """A wall close ahead of the start, so episodes end quickly and often."""
    world = World()
    world.add_segment((config.START_X + 80, config.START_Y - 60),
                      (config.START_X + 80, config.START_Y + 60))
    return world


def test_env_uses_a_given_route():
    route = RouteField()
    env = DrivingEnv(route=route)
    assert env.route is route


def test_one_car_matches_a_bare_env():
    actions = [(i * 7) % config.ACTION_DIM for i in range(300)]

    random.seed(123)
    env = DrivingEnv(crash_world(), random_spawn=False)
    bare = [env.reset()]
    for a in actions:
        state, reward, done = env.step(a)
        bare.append((state, reward, done))
        if done:
            env.reset()

    random.seed(123)
    vec = VecDrivingEnv(crash_world(), 1, use_route=True)
    vecd = [vec.reset()[0]]
    for a in actions:
        states, rewards, dones, _ = vec.step([a])
        vecd.append((states[0], rewards[0], dones[0]))

    assert np.array_equal(bare[0], vecd[0])
    assert sum(d for _, _, d in bare[1:]) >= 1  # the comparison covered an auto-reset
    for (s1, r1, d1), (s2, r2, d2) in zip(bare[1:], vecd[1:]):
        assert np.array_equal(s1, s2) and r1 == r2 and d1 == d2


def test_cars_share_one_route_built_once():
    vec = VecDrivingEnv(World(), 4, use_route=True)
    states = vec.reset()
    assert len(states) == 4 and len(vec.cars) == 4
    assert all(env.route is vec.route for env in vec.envs)
    assert vec.route.builds == 1


def test_changed_wall_rebuilds_the_shared_route_once():
    vec = VecDrivingEnv(World(), 4, use_route=True)
    vec.reset()
    vec.world.add_segment((600, 250), (600, 550))
    vec.step([2] * 4)
    assert vec.route.builds == 2
    assert vec.check_route() is True and vec.route_ok is True


def test_check_route_reports_a_blocked_finish():
    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))
    vec = VecDrivingEnv(world, 3, use_route=True)
    vec.reset()
    assert vec.check_route() is False and vec.route_ok is False


def test_crashing_car_resets_alone_and_is_reported():
    world = World()
    world.add_segment((600, 300), (600, 500))
    vec = VecDrivingEnv(world, 2, use_route=True)
    vec.reset()
    for _ in range(4):
        vec.step([2, 2])
    crasher = vec.envs[1].car
    crasher.x, crasher.y, crasher.angle = 590.0, 400.0, 0.0

    states, rewards, dones, finished = vec.step([2, 2])

    assert dones == [False, True]
    assert rewards[1] == config.R_CRASH
    assert finished == [{"car": 1, "outcome": "crash", "score": 5}]
    assert vec.envs[1].steps == 0 and vec.envs[1].outcome == ""
    assert (vec.cars[1].x, vec.cars[1].y) == (config.START_X, config.START_Y)
    assert vec.envs[0].steps == 5 and vec.envs[0].outcome == ""


def test_terminal_transition_keeps_the_terminal_state():
    world = World()
    world.add_segment((600, 300), (600, 500))
    vec = VecDrivingEnv(world, 2, use_route=True)
    vec.reset()
    crasher = vec.envs[1].car
    crasher.x, crasher.y, crasher.angle = 590.0, 400.0, 0.0

    states, _, dones, _ = vec.step([2, 2])

    assert dones[1] is True
    assert states[1][2] < 0.2               # front sensor: the wall is right there
    assert vec.states[1][2] == 1.0          # the next episode starts on open road
    assert not np.array_equal(states[1], vec.states[1])
    assert np.array_equal(states[0], vec.states[0])
