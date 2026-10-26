"""Car state and physics. Used identically by the human and the AI."""
import math

import pygame

from car_sim import config


class Car:
    def __init__(self):
        self.reset()

    def reset(self, x=None, y=None, angle=0.0):
        self.x = float(config.START_X if x is None else x)
        self.y = float(config.START_Y if y is None else y)
        self.angle = float(angle)  # radians; 0 faces right
        self.speed = 0.0
        self.ang_vel = 0.0

    def step(self, dt, throttle, brake, steer):
        """Advance one frame. Returns the distance moved this frame."""
        dt = min(dt, config.MAX_DT)

        # Speed: throttle accelerates, brake is stronger, friction only when coasting.
        accel = throttle * config.MAX_ACCEL
        braking = brake * config.MAX_ACCEL * config.BRAKE_FORCE
        self.speed += (accel - braking) * dt
        if throttle == 0:
            self.speed -= config.FRICTION * dt
        self.speed = max(0.0, min(config.MAX_SPEED, self.speed))

        # Steering: ease the angular velocity toward the target instead of snapping.
        target = steer * config.MAX_STEER_RATE
        self.ang_vel += (target - self.ang_vel) * config.STEER_SMOOTHING * dt
        self.angle += self.ang_vel * dt

        # Position: move along the heading.
        old_x, old_y = self.x, self.y
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt
        return math.hypot(self.x - old_x, self.y - old_y)

    def draw(self, surface, color, alpha=255):
        """alpha below 255 draws the car see-through (the extra cars in multi-car training)."""
        body = pygame.Surface((config.CAR_W, config.CAR_H), pygame.SRCALPHA)
        pygame.draw.rect(body, color, (0, 0, config.CAR_W, config.CAR_H))
        # A lighter strip at the front shows which way the car faces.
        front = tuple(min(255, c + 70) for c in color)
        pygame.draw.rect(body, front, (config.CAR_W - 10, 2, 6, config.CAR_H - 4))
        # Pygame's y axis points down, so negate the angle to rotate the right way.
        rotated = pygame.transform.rotate(body, -math.degrees(self.angle))
        if alpha < 255:
            rotated.set_alpha(alpha)
        surface.blit(rotated, rotated.get_rect(center=(int(self.x), int(self.y))))
