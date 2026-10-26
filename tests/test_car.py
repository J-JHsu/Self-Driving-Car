import math

import pygame
import pytest

from car_sim import config
from car_sim.car import Car

DT = 1 / 60


def run(car, seconds, throttle=0.0, brake=0.0, steer=0.0):
    for _ in range(int(seconds / DT)):
        car.step(DT, throttle, brake, steer)


def test_accelerates():
    car = Car()
    run(car, 1.0, throttle=1.0)
    assert 0 < car.speed <= config.MAX_SPEED


def test_speed_clamped():
    car = Car()
    run(car, 10.0, throttle=1.0)
    assert car.speed == pytest.approx(config.MAX_SPEED)


def test_friction_slows_and_never_goes_negative():
    car = Car()
    car.speed = 100.0
    run(car, 0.5)
    assert 0 < car.speed < 100.0
    run(car, 10.0)
    assert car.speed == 0.0


def test_brake_stronger_than_friction():
    braking, coasting = Car(), Car()
    braking.speed = coasting.speed = 150.0
    run(braking, 0.3, brake=1.0)
    run(coasting, 0.3)
    assert braking.speed < coasting.speed


def test_steering_is_smooth():
    car = Car()
    car.step(DT, 0.0, 0.0, 1.0)
    assert 0 < car.ang_vel < config.MAX_STEER_RATE
    run(car, 1.0, steer=1.0)
    assert car.ang_vel > 0.95 * config.MAX_STEER_RATE


def test_returns_distance_moved():
    car = Car()
    car.speed = 100.0
    x0, y0 = car.x, car.y
    moved = car.step(0.01, 1.0, 0.0, 0.0)
    assert moved == pytest.approx(math.hypot(car.x - x0, car.y - y0))
    assert moved > 0


def test_dt_clamped():
    lagged, normal = Car(), Car()
    lagged.step(10.0, 1.0, 0.0, 0.0)
    normal.step(config.MAX_DT, 1.0, 0.0, 0.0)
    assert lagged.x == pytest.approx(normal.x)
    assert lagged.speed == pytest.approx(normal.speed)


def test_reset():
    car = Car()
    run(car, 1.0, throttle=1.0, steer=1.0)
    car.reset()
    assert (car.x, car.y) == (config.START_X, config.START_Y)
    assert car.angle == 0.0 and car.speed == 0.0 and car.ang_vel == 0.0


def test_draw_marks_pixels_at_car_center():
    surface = pygame.Surface((config.GAME_W, config.GAME_H))
    surface.fill((0, 0, 0))
    car = Car()
    car.angle = 0.7
    car.draw(surface, (10, 200, 30))
    assert tuple(surface.get_at((int(car.x), int(car.y))))[:3] == (10, 200, 30)


def test_reset_to_given_position():
    car = Car()
    car.speed = 50.0
    car.ang_vel = 1.0
    car.reset(500, 300, 1.0)
    assert (car.x, car.y, car.angle) == (500, 300, 1.0)
    assert car.speed == 0.0 and car.ang_vel == 0.0
