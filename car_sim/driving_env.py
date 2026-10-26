"""The AI-facing wrapper: turns Car + World into reset() / step(action)."""
import math
import random

import numpy as np

from car_sim import config
from car_sim.car import Car
from car_sim.route import RouteField, StraightField
from car_sim.world import World


def compute_reward(collided, speed_norm, proximity_norm, brake, progress, reached):
    """Score one step.

    speed_norm and proximity_norm are 0-1, brake is 0-1, progress is the pixels of
    route covered this step (negative when moving away from the finish).
    """
    if reached:
        return float(config.R_GOAL)
    if collided:
        return config.R_CRASH
    reward = config.R_PROGRESS * progress  # follow the route
    if speed_norm < config.SLOW_SPEED_NORM:
        reward += config.R_SLOW  # do not just sit still
    if proximity_norm < config.FRONT_NEAR_NORM:
        closeness = (config.FRONT_NEAR_NORM - proximity_norm) / config.FRONT_NEAR_NORM
        reward -= config.R_FRONT_NEAR * closeness
    if speed_norm > config.FAST_SPEED_NORM and proximity_norm < config.FAST_NEAR_NORM:
        reward -= config.R_FAST_NEAR  # slow down when something is close ahead
    return float(reward)


def proximity(state):
    """How close the nearest obstacle feels, 0-1 (lower is closer).

    The front sensor counts as is; a side sensor counts PROXIMITY_SIDE_MARGIN sooner,
    so the car also learns to stay off walls it is driving alongside.
    """
    mid = len(config.SENSOR_ANGLES) // 2
    sides = [state[i] for i in range(len(config.SENSOR_ANGLES)) if i != mid]
    return float(min(state[mid], min(sides) + config.PROXIMITY_SIDE_MARGIN))


def _wrap(angle):
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class DrivingEnv:
    def __init__(self, world=None, random_spawn=None, use_route=True):
        self.world = world if world is not None else World()
        self.random_spawn = config.RANDOM_SPAWN if random_spawn is None else random_spawn
        self.car = Car()
        # use_route=False is the harder 'no route' mode: the car is told only the straight-line
        # direction and distance to the finish and has to learn to get around walls by itself.
        self.route = RouteField() if use_route else StraightField()
        self.route_ok = False  # does the start currently have a route to the finish?
        self.outcome = ""      # "", "goal", "crash", "stall" or "timeout"
        self.steps = 0
        self.stall_steps = 0
        self.last_controls = (0.0, 0.0, 0.0)  # throttle, brake, steer (for the UI)
        self._route_dist = math.inf
        self._route_builds = 0
        self._state = self.get_state()  # reused by step() so sensors are cast once per step

    def check_route(self):
        """Refresh and return whether the start currently has a route to the finish."""
        self.route_ok = self.route.ensure(self.world)
        return self.route_ok

    def get_state(self):
        self.check_route()
        sensors = [d / config.MAX_SENSOR_DISTANCE for d in self.world.sensor_distances(self.car)]
        speed = self.car.speed / config.MAX_SPEED
        heading = self.route.direction(self.car.x, self.car.y)
        distance = self.route.distance(self.car.x, self.car.y)
        if heading is None or not math.isfinite(distance):
            rel_dir, dist_norm = 0.0, 1.0
        else:
            rel_dir = _wrap(heading - self.car.angle) / math.pi
            dist_norm = min(1.0, distance / config.ROUTE_DIST_SCALE)
        return np.array(sensors + [speed, rel_dir, dist_norm], dtype=np.float32)

    def _place_car(self):
        """Random free spot with a route; falls back to the world's start if none is found."""
        m = config.SPAWN_MARGIN
        for _ in range(config.SPAWN_TRIES):
            self.car.x = random.uniform(m, config.GAME_W - m)
            self.car.y = random.uniform(m, config.GAME_H - m)
            self.car.angle = random.uniform(-math.pi, math.pi)
            if not self.world.collided(self.car) and self.route.reachable(self.car.x, self.car.y):
                return
        self.car.reset(*self.world.start)

    def reset(self):
        self.route_ok = self.route.ensure(self.world)
        heading = self.route.direction(*self.world.start) if self.route_ok else None
        angle = 0.0
        if heading is not None:
            angle = heading + random.uniform(-config.START_HEADING_JITTER, config.START_HEADING_JITTER)
        self.car.reset(self.world.start[0], self.world.start[1], angle)
        if self.random_spawn:
            self._place_car()
        self.stall_steps = 0
        self.steps = 0
        self.outcome = ""
        self.last_controls = (0.0, 0.0, 0.0)
        self._state = self.get_state()
        self._route_dist = self.route.distance(self.car.x, self.car.y)
        self._route_builds = self.route.builds
        return self._state

    def step(self, action, dt=1 / 60):
        """Apply a steering action. Returns (next_state, reward, done)."""
        state = self._state
        front = float(state[len(config.SENSOR_ANGLES) // 2])  # center sensor
        speed_norm = float(state[len(config.SENSOR_ANGLES)])

        # Faster car -> gentler steering, so it stays stable at speed.
        steer = config.STEER_MAP[action] * (1.0 - config.STEER_SPEED_SCALE * speed_norm)

        # Brake harder the closer the obstacle straight ahead.
        brake = float(np.clip(1.0 - front / config.BRAKE_TRIGGER_DIST, 0.0, 1.0))
        if brake > config.BRAKE_ACTIVE_THRESHOLD:
            throttle = 0.0
        else:
            brake = 0.0
            throttle = 1.0
            if abs(config.STEER_MAP[action]) > config.SHARP_STEER:
                throttle *= config.SHARP_STEER_THROTTLE
        self.last_controls = (throttle, brake, float(config.STEER_MAP[action]))

        moved = self.car.step(dt, throttle, brake, steer)
        collided = self.world.collided(self.car)
        next_state = self.get_state()
        self._state = next_state
        self.steps += 1

        # Progress along the route; a rebuilt route (the user drew or moved something)
        # changes distances for reasons the car did not cause, so it counts as zero.
        new_dist = self.route.distance(self.car.x, self.car.y)
        rebuilt = self.route.builds != self._route_builds
        if rebuilt or not (math.isfinite(new_dist) and math.isfinite(self._route_dist)):
            progress = 0.0
        else:
            progress = self._route_dist - new_dist
        self._route_dist = new_dist
        self._route_builds = self.route.builds

        goal_x, goal_y = self.world.goal
        reached = math.hypot(self.car.x - goal_x, self.car.y - goal_y) <= config.GOAL_RADIUS
        self.stall_steps = self.stall_steps + 1 if moved < config.STALL_DISTANCE_EPS else 0

        reward = compute_reward(
            collided, float(next_state[len(config.SENSOR_ANGLES)]), proximity(next_state),
            brake, progress, reached,
        )
        done = True
        if reached:
            self.outcome = "goal"
        elif collided:
            self.outcome = "crash"
        elif self.stall_steps >= config.STALL_LIMIT:
            self.outcome = "stall"
            reward = config.R_STALL
        elif self.steps >= config.MAX_EPISODE_STEPS:
            self.outcome = "timeout"
            reward = config.R_TIMEOUT
        else:
            done = False
        return next_state, float(reward), done
