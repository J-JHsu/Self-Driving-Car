"""Drive the car yourself.

Keys: Up/W throttle, Down/S brake, Left/A and Right/D steer,
V toggle sensors, C clear obstacles, R restart after a crash or a win, Space pause,
1 then click = place the start, 2 then click = place the finish (Esc cancels),
P show the shortest route. Left-drag draws walls, right-drag erases them.
"""
import math

import pygame

from car_sim import config, ui
from car_sim.car import Car
from car_sim.route import RouteField
from car_sim.world import World

MOVE_THRESHOLD = 1.0  # pixels per frame needed to earn score


def read_keys(keys):
    """Map held keys to (throttle, brake, steer)."""
    throttle = 1.0 if keys[pygame.K_UP] or keys[pygame.K_w] else 0.0
    brake = 1.0 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0.0
    steer = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        steer -= 1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        steer += 1.0
    return throttle, brake, steer


class HumanGame:
    """Car + world + score for the human-driven mode."""

    def __init__(self):
        self.world = World()
        self.car = Car()
        self.car.reset(*self.world.start)
        self.score = 0.0
        self.max_score = 0.0
        self.game_over = False
        self.won = False
        self.elapsed = 0.0

    def reset(self):
        self.car.reset(*self.world.start)
        self.score = 0.0
        self.game_over = False
        self.won = False
        self.elapsed = 0.0

    def update(self, dt, throttle, brake, steer):
        if self.game_over:
            return
        self.elapsed += dt
        moved = self.car.step(dt, throttle, brake, steer)
        if moved > MOVE_THRESHOLD:
            self.score += moved
            self.max_score = max(self.max_score, self.score)
        gx, gy = self.world.goal
        if math.hypot(self.car.x - gx, self.car.y - gy) <= config.GOAL_RADIUS:
            self.won = True
            self.game_over = True
        elif self.world.collided(self.car):
            self.game_over = True


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.GAME_W, config.GAME_H + config.PANEL_H))
    pygame.display.set_caption("Human Car Game")
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 40)
    clock = pygame.time.Clock()

    game = HumanGame()
    route = RouteField()
    show_sensors = True
    show_route = False
    paused = False
    placing = None          # "start", "goal" or None
    no_draw_until_release = False
    controls = (0.0, 0.0, 0.0)
    running = True

    while running:
        dt = clock.tick(config.FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            before = (game.world.start, game.world.goal)
            new_mode = ui.handle_placement_event(event, game.world, placing)
            if event.type == pygame.MOUSEBUTTONDOWN and placing and new_mode is None:
                no_draw_until_release = True  # this click placed a point; do not draw a wall
            placing = new_mode
            if game.world.start != before[0]:
                game.reset()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_p:
                    show_route = not show_route
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    game.world.clear()
                elif event.key == pygame.K_r and game.game_over:
                    game.reset()

        left, _, right = pygame.mouse.get_pressed()[:3]
        if not left:
            no_draw_until_release = False
        free_hand = placing is None
        pressed = left and not game.game_over and free_hand and not no_draw_until_release
        game.world.handle_mouse(pygame.mouse.get_pos(), pressed)
        game.world.handle_erase(pygame.mouse.get_pos(), right and free_hand)

        controls = (0.0, 0.0, 0.0) if game.game_over or paused else read_keys(pygame.key.get_pressed())
        if not paused:
            game.update(dt, *controls)

        screen.fill(config.BG_COLOR, (0, 0, config.GAME_W, config.GAME_H))
        game.world.draw(screen)
        has_route = route.ensure(game.world)
        if show_route and has_route:
            ui.draw_route(screen, route.path(game.car.x, game.car.y))
        ui.draw_markers(screen, game.world, placing)
        if show_sensors and not game.game_over:
            ui.draw_sensors(screen, game.world, game.car)
        game.car.draw(screen, config.HUMAN_CAR_COLOR)

        if placing:
            hint = f"Click the map to place the {'START' if placing == 'start' else 'FINISH'}  (Esc cancels)"
        else:
            hint = ui.HINT_PLACE
        lines = [
            f"Score: {int(game.score)}   Max score: {int(game.max_score)}   Time: {game.elapsed:.1f}s",
            ui.HINT_DRIVE,
            ui.HINT_DRAW,
            hint,
            ui.HINT_VIEW,
        ]
        if not has_route:
            lines.append("No route from start to finish: move a point or open a gap")
        ui.draw_panel(screen, font, lines)
        if game.won:
            ui.draw_status(screen, big_font, font, "YOU WIN!",
                           f"{game.elapsed:.1f}s - press R to restart", config.START_MARKER_COLOR)
        elif game.game_over:
            ui.draw_status(screen, big_font, font, "GAME OVER", "press R to restart",
                           config.GAME_OVER_COLOR)
        elif paused:
            ui.draw_status(screen, big_font, font, "PAUSED", "press Space to resume",
                           config.TEXT_COLOR)
        ui.draw_steering_wheel(screen, controls[2])
        ui.draw_pedal(screen, ui.THROTTLE_RECT, controls[0], config.THROTTLE_COLOR, "Gas", font)
        ui.draw_pedal(screen, ui.BRAKE_RECT, controls[1], config.BRAKE_COLOR, "Brake", font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
