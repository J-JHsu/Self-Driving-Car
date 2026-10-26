"""Drawing helpers shared by the human and AI programs."""
import math

import pygame

from car_sim import config

PANEL_TOP = config.GAME_H
# Text fills the left side of the panel; the wheel and pedals sit to its right and the
# big status message (YOU WIN / GAME OVER / PAUSED) in the last corner. tests/test_ui.py
# checks that none of these overlap.
WHEEL_CENTER = (860, PANEL_TOP + 85)
WHEEL_RADIUS = 45
THROTTLE_RECT = (960, PANEL_TOP + 45, 40, 100)
BRAKE_RECT = (1015, PANEL_TOP + 45, 40, 100)
MESSAGE_POS = (1090, PANEL_TOP + 50)
MAX_PANEL_LINES = 7
PEDAL_BG = (60, 60, 70)

HINT_DRAW = "Mouse: left-drag draws walls, right-drag erases"
HINT_PLACE = "1: place start    2: place finish    Esc: cancel"
HINT_VIEW = "P: route    V: sensors    C: clear    Space: pause"
HINT_VIEW_AI = HINT_VIEW + "    F: fast"
HINT_DRIVE = "Up/W gas   Down/S brake   Left/Right or A/D steer   R: restart"
ALL_HINTS = (HINT_DRAW, HINT_PLACE, HINT_VIEW, HINT_VIEW_AI, HINT_DRIVE)


def draw_text(surface, font, text, pos, color=config.TEXT_COLOR):
    surface.blit(font.render(text, True, color), pos)


def draw_status(surface, big_font, font, title, subtitle, color):
    """Big message in the panel's right-hand corner, e.g. PAUSED / press Space."""
    x, y = MESSAGE_POS
    draw_text(surface, big_font, title, (x, y), color)
    draw_text(surface, font, subtitle, (x, y + 36), color)


def draw_panel(surface, font, lines):
    """Fill the bottom panel and write one line of text per entry."""
    pygame.draw.rect(surface, config.PANEL_COLOR, (0, PANEL_TOP, config.GAME_W, config.PANEL_H))
    for i, line in enumerate(lines):
        draw_text(surface, font, line, (15, PANEL_TOP + 10 + i * 22))


def draw_steering_wheel(surface, steer):
    """steer in [-1, 1]: the indicator line swings up to 90 degrees each way."""
    cx, cy = WHEEL_CENTER
    pygame.draw.circle(surface, config.WHEEL_COLOR, WHEEL_CENTER, WHEEL_RADIUS, 4)
    angle = steer * math.pi / 2
    end = (int(cx + math.sin(angle) * WHEEL_RADIUS), int(cy - math.cos(angle) * WHEEL_RADIUS))
    pygame.draw.line(surface, config.WHEEL_COLOR, WHEEL_CENTER, end, 4)


def draw_pedal(surface, rect, value, color, label, font):
    """A vertical bar that fills from the bottom as value goes from 0 to 1."""
    x, y, w, h = rect
    value = max(0.0, min(1.0, value))
    pygame.draw.rect(surface, PEDAL_BG, rect)
    fill_h = int(h * value)
    if fill_h > 0:
        pygame.draw.rect(surface, color, (x, y + h - fill_h, w, fill_h))
    pygame.draw.rect(surface, config.WHEEL_COLOR, rect, 2)
    draw_text(surface, font, label, (x - 4, y - 24))


def draw_sensors(surface, world, car):
    """Draw each sensor ray and a dot where it ends."""
    for offset in config.SENSOR_ANGLES:
        angle = car.angle + offset
        dist = world.cast_ray(car.x, car.y, angle)
        end = (int(car.x + math.cos(angle) * dist), int(car.y + math.sin(angle) * dist))
        pygame.draw.line(surface, config.SENSOR_COLOR, (int(car.x), int(car.y)), end, 1)
        pygame.draw.circle(surface, config.SENSOR_COLOR, end, 4)


def draw_markers(surface, world, placing):
    """Draw the start (green dot) and finish (red dot inside its goal circle).

    `placing` is "start", "goal" or None; the point being placed gets a white ring.
    """
    sx, sy = int(world.start[0]), int(world.start[1])
    gx, gy = int(world.goal[0]), int(world.goal[1])
    pygame.draw.circle(surface, config.GOAL_MARKER_COLOR, (gx, gy), config.GOAL_RADIUS, 2)
    pygame.draw.circle(surface, config.GOAL_MARKER_COLOR, (gx, gy), 8)
    pygame.draw.circle(surface, config.START_MARKER_COLOR, (sx, sy), 8)
    if placing == "start":
        pygame.draw.circle(surface, config.TEXT_COLOR, (sx, sy), 16, 2)
    elif placing == "goal":
        pygame.draw.circle(surface, config.TEXT_COLOR, (gx, gy), config.GOAL_RADIUS + 6, 2)


def draw_route(surface, points):
    """Draw the route as a polyline (nothing if there are fewer than two points)."""
    if len(points) >= 2:
        pygame.draw.lines(surface, config.ROUTE_COLOR, False,
                          [(int(x), int(y)) for x, y in points], 2)


def handle_placement_event(event, world, mode):
    """Keys 1/2 pick what to place, a click in the map places it, Esc cancels.

    Returns the new mode ("start", "goal" or None).
    """
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_1:
            return "start"
        if event.key == pygame.K_2:
            return "goal"
        if event.key == pygame.K_ESCAPE:
            return None
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and mode:
        x, y = event.pos
        if 0 <= x < config.GAME_W and 0 <= y < config.GAME_H:
            if mode == "start":
                world.set_start((x, y))
            else:
                world.set_goal((x, y))
            return None
    return mode
