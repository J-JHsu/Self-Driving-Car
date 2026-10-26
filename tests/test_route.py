import math
import random
import time

import pytest

from car_sim import config
from car_sim.route import RouteField, StraightField
from car_sim.world import World, point_segment_distance


def field_for(world):
    field = RouteField()
    field.ensure(world)
    return field


def test_open_map_direction_and_distance():
    field = field_for(World())
    assert field.distance(100, 400) == pytest.approx(1200, rel=0.05)
    assert abs(field.direction(100, 400)) < 0.2


def test_u_pocket_routes_out_the_open_side():
    world = World()
    world.add_segment((600, 300), (600, 500))  # back wall of the pocket
    world.add_segment((400, 300), (600, 300))
    world.add_segment((400, 500), (600, 500))  # pocket opens to the left
    field = field_for(world)
    assert abs(field.direction(500, 400)) > math.pi / 2  # must back out first
    assert field.distance(500, 400) > 1000


def test_walled_off_goal_is_unreachable():
    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))
    field = RouteField()
    assert field.ensure(world) is False
    assert field.distance(100, 400) == math.inf
    assert field.reachable(100, 400) is False
    assert field.direction(100, 400) is None


def test_narrow_gap_blocked_wide_gap_open():
    narrow = World()
    narrow.add_segment((700, 0), (700, 385))
    narrow.add_segment((700, 415), (700, config.GAME_H))
    assert RouteField().ensure(narrow) is False  # 30 px gap: the car cannot fit
    wide = World()
    wide.add_segment((700, 0), (700, 360))
    wide.add_segment((700, 440), (700, config.GAME_H))
    assert RouteField().ensure(wide) is True  # 80 px gap


def test_ensure_rebuilds_only_on_change():
    world = World()
    field = RouteField()
    field.ensure(world)
    field.ensure(world)
    assert field.builds == 1
    world.add_segment((300, 100), (300, 200))
    field.ensure(world)
    assert field.builds == 2
    world.set_goal((900, 500))
    field.ensure(world)
    assert field.builds == 3


def test_point_in_blocked_cell_uses_nearest_free():
    world = World()
    world.add_segment((200, 400), (800, 400))
    field = field_for(world)
    assert math.isfinite(field.distance(500, 405))


def test_path_follows_route_to_goal():
    field = field_for(World())
    pts = field.path(100, 400)
    assert math.dist(pts[0], (100, 400)) < config.GRID_CELL * 2
    assert math.dist(pts[-1], (config.GOAL_X, config.GOAL_Y)) <= config.GOAL_RADIUS
    assert len(pts) <= 400
    assert len(field.path(100, 400, max_points=20)) <= 20


def test_path_empty_when_unreachable():
    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))
    assert field_for(world).path(100, 400) == []


def test_distance_shrinks_smoothly_when_moving_toward_goal():
    field = field_for(World())
    values = [field.distance(100 + i, 400) for i in range(0, 40)]
    steps = [a - b for a, b in zip(values, values[1:])]
    assert all(0 < s < 3 for s in steps)  # about 1 px per px, no 10 px jumps


def test_build_is_fast():
    rng = random.Random(0)
    world = World()
    for _ in range(200):
        x, y = rng.uniform(0, 1400), rng.uniform(0, 800)
        world.add_segment((x, y), (x + rng.uniform(-60, 60), y + rng.uniform(-60, 60)))
    start = time.perf_counter()
    RouteField().ensure(world)
    assert time.perf_counter() - start < 0.25


def test_finish_touching_a_wall_still_has_a_route():
    world = World()
    world.add_segment((1250, 400), (1350, 400))  # the finish sits right on a wall
    field = RouteField()
    assert field.ensure(world) is True
    assert math.isfinite(field.distance(100, 400))
    assert field.path(100, 400)[-1] == (config.GOAL_X, config.GOAL_Y)


def test_finish_in_map_corner_still_has_a_route():
    world = World()
    world.set_goal((5, 5))
    assert RouteField().ensure(world) is True


def test_finish_fully_enclosed_is_still_unreachable():
    world = World()
    for a, b in [((1200, 300), (1400, 300)), ((1200, 500), (1400, 500)), ((1200, 300), (1200, 500))]:
        world.add_segment(a, b)
    assert RouteField().ensure(world) is False


def test_cache_does_not_confuse_a_new_world_reusing_an_old_id():
    field = RouteField()
    blocked = World()
    blocked.add_segment((1200, 0), (1200, config.GAME_H))
    assert field.ensure(blocked) is False
    del blocked
    for _ in range(200):  # new worlds may land on the freed object's id
        fresh = World()
        fresh.add_segment((300, 100), (300, 110))
        assert field.ensure(fresh) is True
    assert field._world is fresh  # the field keeps the world it built for, so ids cannot be reused


def test_route_keeps_a_margin_from_walls():
    world = World()
    wall = ((700, 200), (700, 800))  # a wall the route must go around, at the top
    world.add_segment(*wall)
    field = field_for(world)
    points = field.path(100, 600)
    assert len(points) > 10
    for p in points:
        assert point_segment_distance(p, *wall) >= config.WALL_CLEARANCE - 1, p


def test_straight_field_ignores_walls():
    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))  # would block a real route
    field = StraightField()
    assert field.ensure(world) is True
    assert field.reachable(100, 400)
    assert field.distance(100, 400) == pytest.approx(1200)
    assert field.direction(100, 400) == pytest.approx(0.0)
    assert field.direction(1300, 600) == pytest.approx(-math.pi / 2)
    assert field.path(100, 400) == [(100.0, 400.0), (1300.0, 400.0)]


def test_straight_field_builds_only_when_finish_moves():
    world = World()
    field = StraightField()
    field.ensure(world)
    field.ensure(world)
    world.add_segment((300, 100), (300, 200))
    field.ensure(world)
    assert field.builds == 1
    world.set_goal((900, 500))
    field.ensure(world)
    assert field.builds == 2


def test_tight_gap_is_still_passable():
    world = World()
    world.add_segment((700, 0), (700, 370))
    world.add_segment((700, 430), (700, config.GAME_H))  # a 60 px gap: tight but the car fits
    assert RouteField().ensure(world) is True


def test_route_runs_down_the_middle_of_a_corridor():
    world = World()
    world.set_goal((1300, 330))  # the shortest line would cut towards the top wall
    walls = [((300, 350), (1000, 350)), ((300, 450), (1000, 450))]  # a 100 px wide corridor
    for wall in walls:
        world.add_segment(*wall)
    points = [p for p in field_for(world).path(100, 400) if 300 <= p[0] <= 1000]
    assert points
    for p in points:
        assert min(point_segment_distance(p, *w) for w in walls) >= 38, p
