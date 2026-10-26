from types import SimpleNamespace

import pytest

from car_sim import config
from car_sim.world import World, point_segment_distance


def car_at(x, y, angle=0.0):
    return SimpleNamespace(x=x, y=y, angle=angle)


def test_distance_projection_inside():
    assert point_segment_distance((5, 3), (0, 0), (10, 0)) == pytest.approx(3.0)


def test_distance_before_a():
    assert point_segment_distance((-3, 4), (0, 0), (10, 0)) == pytest.approx(5.0)


def test_distance_past_b():
    assert point_segment_distance((13, 4), (0, 0), (10, 0)) == pytest.approx(5.0)


def test_zero_length_segment():
    assert point_segment_distance((5, 6), (2, 2), (2, 2)) == pytest.approx(5.0)


def test_ray_hits_wall():
    d = World().cast_ray(1300, 300, 0.0)
    assert 90 < d <= 100


def test_ray_hits_obstacle():
    world = World()
    world.add_segment((300, 200), (300, 400))
    d = world.cast_ray(200, 300, 0.0)
    assert abs(d - 100) <= config.SENSOR_STEP


def test_ray_clear_returns_max():
    assert World().cast_ray(100, 300, 0.0) == config.MAX_SENSOR_DISTANCE


def test_sensor_distances_one_per_angle_relative_to_heading():
    world = World()
    world.add_segment((300, 200), (300, 400))
    ahead = world.sensor_distances(car_at(200, 300, 0.0))
    assert len(ahead) == len(config.SENSOR_ANGLES)
    assert ahead[2] < 110  # center ray sees the wall
    away = world.sensor_distances(car_at(200, 300, 3.14159))
    assert away[2] == config.MAX_SENSOR_DISTANCE or away[2] > ahead[2]


def test_collision_edge():
    assert World().collided(car_at(5, 300))


def test_collision_obstacle():
    world = World()
    world.add_segment((300, 200), (300, 400))
    assert world.collided(car_at(310, 300))


def test_no_collision_open_space():
    assert not World().collided(car_at(400, 300))


def test_click_without_drag_is_safe():
    world = World()
    world.handle_mouse((100, 100), True)
    world.handle_mouse((100, 100), True)
    world.handle_mouse((100, 100), False)
    assert world.cast_ray(50, 100, 0.0) >= 0
    assert world.collided(car_at(105, 100))  # behaves as a point
    assert not world.collided(car_at(400, 300))


def test_mouse_drag_in_panel_adds_nothing():
    world = World()
    world.handle_mouse((100, config.GAME_H + 10), True)
    world.handle_mouse((120, config.GAME_H + 20), True)
    assert world.obstacles == []


def test_mouse_drag_connects_segments_and_release_resets():
    world = World()
    world.handle_mouse((100, 100), True)
    world.handle_mouse((110, 100), True)
    world.handle_mouse((120, 100), True)
    assert len(world.obstacles) == 2
    world.handle_mouse((120, 100), False)
    world.handle_mouse((300, 300), True)
    assert len(world.obstacles) == 2  # new stroke does not connect to the old one


def test_holding_mouse_still_adds_only_one_point():
    world = World()
    for _ in range(30):
        world.handle_mouse((100, 100), True)
    assert world.obstacles == []  # nothing yet: the stroke has not moved
    world.handle_mouse((100, 100), False)
    assert len(world.obstacles) == 1  # a click leaves one dot


def test_mouse_jitter_below_threshold_is_ignored():
    world = World()
    for pos in [(100, 100), (101, 100), (100, 101), (101, 101)]:
        world.handle_mouse(pos, True)
    world.handle_mouse((101, 101), False)
    assert len(world.obstacles) == 1


def test_clear_then_add_same_count_invalidates_sensor_cache():
    world = World()
    world.add_segment((300, 200), (300, 400))
    assert world.cast_ray(200, 300, 0.0) < 120  # builds the cache
    world.clear()
    world.add_segment((600, 200), (600, 400))
    assert world.cast_ray(200, 300, 0.0) == config.MAX_SENSOR_DISTANCE


def test_clear_removes_obstacles():
    world = World()
    world.add_segment((0, 0), (10, 10))
    world.clear()
    assert world.obstacles == []


def test_defaults_are_config_values():
    world = World()
    assert world.start == (config.START_X, config.START_Y)
    assert world.goal == (config.GOAL_X, config.GOAL_Y)
    assert world.version == 0


def test_set_start_and_goal_bump_version():
    world = World()
    world.set_start((200, 300))
    assert world.start == (200, 300)
    assert world.version == 1
    world.set_goal((900, 500))
    assert world.goal == (900, 500)
    assert world.version == 2


def test_out_of_bounds_points_ignored():
    world = World()
    start, goal = world.start, world.goal
    world.set_start((1500, 10))
    world.set_goal((-5, 20))
    world.set_goal((100, config.GAME_H + 5))  # the panel area is not part of the map
    assert (world.start, world.goal) == (start, goal)
    assert world.version == 0


def test_add_segment_and_clear_bump_version():
    world = World()
    world.add_segment((0, 0), (10, 10))
    assert world.version == 1
    world.clear()
    assert world.version == 2


def test_clear_keeps_start_and_goal():
    world = World()
    world.set_start((200, 300))
    world.set_goal((900, 500))
    world.clear()
    assert world.start == (200, 300)
    assert world.goal == (900, 500)


def test_erase_removes_only_nearby_segments_and_bumps_version():
    world = World()
    world.add_segment((100, 100), (200, 100))
    world.add_segment((100, 300), (200, 300))
    version = world.version
    assert world.erase_near((150, 105), 12) == 1
    assert world.obstacles == [((100, 300), (200, 300))]
    assert world.version == version + 1


def test_erase_nothing_nearby_changes_nothing():
    world = World()
    world.add_segment((100, 100), (200, 100))
    version = world.version
    assert world.erase_near((150, 200), 12) == 0
    assert len(world.obstacles) == 1 and world.version == version


def test_erased_wall_no_longer_blocks_rays():
    world = World()
    world.add_segment((300, 200), (300, 400))
    assert world.cast_ray(200, 300, 0.0) < 120  # builds the cache
    world.erase_near((300, 300), 12)
    assert world.cast_ray(200, 300, 0.0) == config.MAX_SENSOR_DISTANCE


def test_handle_erase_only_inside_game_area():
    world = World()
    world.add_segment((100, 100), (200, 100))
    world.handle_erase((150, config.GAME_H + 20), True)  # in the panel
    world.handle_erase((150, 100), False)                # button not held
    assert len(world.obstacles) == 1
    world.handle_erase((150, 100), True)
    assert world.obstacles == []
