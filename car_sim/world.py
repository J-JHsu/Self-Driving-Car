"""Obstacles, sensor rays and collision checks."""
import math

import numpy as np
import pygame

from car_sim import config


def point_segment_distance(p, a, b):
    """Shortest distance from point p to the segment a-b."""
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    len2 = abx * abx + aby * aby
    if len2 == 0:  # a and b are the same point
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / len2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


class World:
    def __init__(self):
        self.obstacles = []  # list of ((x1, y1), (x2, y2))
        self.start = (config.START_X, config.START_Y)
        self.goal = (config.GOAL_X, config.GOAL_Y)
        self.version = 0  # bumps whenever walls, start or goal change
        self._prev_mouse = None
        self._stroke_drew = False
        self._seg_cache = None
        self._seg_cache_len = -1

    # --- obstacles ---------------------------------------------------------
    def add_segment(self, a, b):
        self.obstacles.append((tuple(a), tuple(b)))
        self._seg_cache_len = -1  # force the numpy cache to rebuild
        self.version += 1

    def clear(self):
        self.obstacles.clear()
        self._prev_mouse = None
        self._stroke_drew = False
        self._seg_cache_len = -1
        self.version += 1  # start and goal are kept

    @staticmethod
    def _in_map(pos):
        return 0 <= pos[0] < config.GAME_W and 0 <= pos[1] < config.GAME_H

    def set_start(self, pos):
        if self._in_map(pos):
            self.start = (pos[0], pos[1])
            self.version += 1

    def set_goal(self, pos):
        if self._in_map(pos):
            self.goal = (pos[0], pos[1])
            self.version += 1

    def handle_mouse(self, pos, pressed):
        """Draw while the left button is held in the game area.

        A segment is added only once the mouse has moved at least MOUSE_MIN_SEGMENT
        pixels, so holding still does not pile up segments. A click that never moves
        leaves a single dot when the button is released.
        """
        inside = 0 <= pos[0] < config.GAME_W and 0 <= pos[1] < config.GAME_H
        if pressed and inside:
            if self._prev_mouse is None:
                self._prev_mouse = tuple(pos)
                self._stroke_drew = False
            elif math.dist(self._prev_mouse, pos) >= config.MOUSE_MIN_SEGMENT:
                self.add_segment(self._prev_mouse, pos)
                self._prev_mouse = tuple(pos)
                self._stroke_drew = True
        else:
            if self._prev_mouse is not None and not self._stroke_drew:
                self.add_segment(self._prev_mouse, self._prev_mouse)
            self._prev_mouse = None

    def erase_near(self, pos, radius=None):
        """Delete every wall segment within `radius` of pos. Returns how many were removed."""
        radius = config.ERASE_RADIUS if radius is None else radius
        keep = [seg for seg in self.obstacles if point_segment_distance(pos, *seg) > radius]
        removed = len(self.obstacles) - len(keep)
        if removed:
            self.obstacles[:] = keep
            self._seg_cache_len = -1
            self.version += 1
        return removed

    def handle_erase(self, pos, pressed):
        """Erase while the right button is held in the game area."""
        if pressed and 0 <= pos[0] < config.GAME_W and 0 <= pos[1] < config.GAME_H:
            self.erase_near(pos)

    def _segments_array(self):
        if self._seg_cache_len != len(self.obstacles):
            self._seg_cache = np.array(
                [[a[0], a[1], b[0], b[1]] for a, b in self.obstacles], dtype=np.float64
            ).reshape(-1, 4)
            self._seg_cache_len = len(self.obstacles)
        return self._seg_cache

    # --- sensors -----------------------------------------------------------
    def cast_ray(self, x, y, angle):
        """Distance in pixels to the nearest wall or obstacle (at most MAX_SENSOR_DISTANCE)."""
        ds = np.arange(0, config.MAX_SENSOR_DISTANCE, config.SENSOR_STEP, dtype=np.float64)
        px = x + math.cos(angle) * ds
        py = y + math.sin(angle) * ds
        stop = (px < 0) | (px >= config.GAME_W) | (py < 0) | (py >= config.GAME_H)

        segs = self._segments_array()
        if len(segs):
            a = segs[:, :2]
            ab = segs[:, 2:] - a
            len2 = (ab ** 2).sum(axis=1)
            safe_len2 = np.where(len2 == 0, 1.0, len2)
            pts = np.stack([px, py], axis=1)[:, None, :]          # (P, 1, 2)
            ap = pts - a[None, :, :]                               # (P, N, 2)
            t = np.clip((ap * ab[None]).sum(axis=2) / safe_len2, 0.0, 1.0)
            t = np.where(len2[None, :] == 0, 0.0, t)
            closest = a[None] + t[:, :, None] * ab[None]
            dist = np.hypot(*(pts - closest).transpose(2, 0, 1))
            stop |= dist.min(axis=1) < config.SENSOR_HIT_DIST

        if stop.any():
            return float(ds[int(np.argmax(stop))])
        return float(config.MAX_SENSOR_DISTANCE)

    def sensor_distances(self, car):
        return [self.cast_ray(car.x, car.y, car.angle + a) for a in config.SENSOR_ANGLES]

    # --- collisions --------------------------------------------------------
    def collided(self, car):
        half_w = config.CAR_W / 2
        half_h = config.CAR_H / 2
        if (
            car.x < half_w
            or car.x > config.GAME_W - half_w
            or car.y < half_h
            or car.y > config.GAME_H - half_h
        ):
            return True
        center = (car.x, car.y)
        return any(
            point_segment_distance(center, a, b) < config.CAR_W / 2 for a, b in self.obstacles
        )

    # --- rendering ---------------------------------------------------------
    def draw(self, surface):
        for a, b in self.obstacles:
            pygame.draw.line(
                surface,
                config.OBSTACLE_COLOR,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                3,
            )
