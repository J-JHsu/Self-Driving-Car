"""Watch a Double DQN agent learn to drive, live.

Mouse: left-drag draws walls, right-drag erases them.
Keys: 1 then click = place the start, 2 then click = place the finish (Esc cancels),
P show the shortest route, V toggle sensors, C clear obstacles, Space pause,
F toggle fast training (uncapped speed).
The model saves to model.pt every few episodes and when you close the window,
and is loaded automatically next time.

`python self_driving_car_ai.py --no-route` is the harder mode: the car is told only the
straight-line direction and distance to the finish, so it must learn to get around walls by
crashing into them. It uses its own file, model_no_route.pt.
"""
import sys
import os
from collections import deque

import pygame

from car_sim import config, ui
from car_sim.driving_env import DrivingEnv
from car_sim.dqn import DoubleDQNAgent

NO_ROUTE_MESSAGE = "No route from start to finish: move a point or open a gap"


def train(headless=False, max_steps=None, model_path=None, world=None, use_route=True):
    """Run the training loop.

    Returns {"episodes", "steps", "max_score", "goals", "no_route"}. max_steps counts
    steps in this run only; episodes and max_score carry over from a loaded model.
    headless=True skips the window; if the start has no route it then returns at once
    (no_route True), because nobody is there to fix the map.
    """
    if model_path is None:
        model_path = config.MODEL_PATH if use_route else config.MODEL_PATH_NO_ROUTE
    env = DrivingEnv(world, use_route=use_route)
    agent = DoubleDQNAgent()
    epsilon = config.EPS_START
    episode = 0
    max_score = 0.0

    if os.path.exists(model_path):
        try:
            meta = agent.load(model_path)
            epsilon = float(meta.get("epsilon", epsilon))
            episode = int(meta.get("episode", 0))
            max_score = float(meta.get("max_score", 0.0))
            print(f"Loaded model from {model_path} (episode {episode}, epsilon {epsilon:.3f})")
        except ValueError as exc:
            print(f"{exc} - starting fresh with a new model.")

    screen = font = clock = None
    if not headless:
        pygame.init()
        screen = pygame.display.set_mode((config.GAME_W, config.GAME_H + config.PANEL_H))
        pygame.display.set_caption("Self Driving Car AI")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()

    show_sensors = True
    show_route = False
    fast = False
    placing = None
    no_draw_until_release = False
    recent = deque(maxlen=20)
    recent_goals = deque(maxlen=20)
    goals = 0
    ep_score = 0
    steps = 0
    avg_score = 0.0
    no_route = False
    paused = False        # no route from start to finish
    user_paused = False   # the player pressed Space
    state = env.reset()
    running = True

    def save():
        agent.save(model_path, {"epsilon": epsilon, "episode": episode, "max_score": max_score})

    try:
        while running and (max_steps is None or steps < max_steps):
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        continue
                    before_start = env.world.start
                    new_mode = ui.handle_placement_event(event, env.world, placing)
                    if event.type == pygame.MOUSEBUTTONDOWN and placing and new_mode is None:
                        no_draw_until_release = True  # that click placed a point, not a wall
                    placing = new_mode
                    if env.world.start != before_start:
                        state = env.reset()
                        ep_score = 0
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_v:
                            show_sensors = not show_sensors
                        elif event.key == pygame.K_p:
                            show_route = not show_route
                        elif event.key == pygame.K_SPACE:
                            user_paused = not user_paused
                        elif event.key == pygame.K_c:
                            env.world.clear()
                        elif event.key == pygame.K_f:
                            fast = not fast
                held, _, right = pygame.mouse.get_pressed()[:3]
                if not held:
                    no_draw_until_release = False
                env.world.handle_mouse(
                    pygame.mouse.get_pos(), held and placing is None and not no_draw_until_release
                )
                env.world.handle_erase(pygame.mouse.get_pos(), right and placing is None)
                if not running:
                    break
                if user_paused:
                    _draw(screen, font, env, show_sensors, show_route, placing, fast, episode,
                          ep_score, max_score, avg_score, epsilon, recent_goals, paused, True)
                    clock.tick(config.FPS)
                    continue

            # Nothing to learn from while the finish cannot be reached.
            if not env.check_route():
                if headless:
                    print(NO_ROUTE_MESSAGE)
                    no_route = True
                    break
                paused = True
                _draw(screen, font, env, show_sensors, show_route, placing, fast, episode,
                      ep_score, max_score, avg_score, epsilon, recent_goals, paused, user_paused)
                clock.tick(config.FPS)
                continue
            if paused:  # the map was fixed: start a clean episode
                paused = False
                state = env.reset()
                ep_score = 0

            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            ep_score += 1
            steps += 1
            epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)
            if steps % config.TARGET_UPDATE_STEPS == 0:
                agent.update_target()

            if done:
                episode += 1
                reached = env.outcome == "goal"
                goals += int(reached)
                recent_goals.append(reached)
                max_score = max(max_score, ep_score)
                recent.append(ep_score)
                avg_score = sum(recent) / len(recent)
                print(
                    f"Episode {episode} | {env.outcome} | score {ep_score} | max {int(max_score)} "
                    f"| avg20 {avg_score:.1f} | goals {goals} | epsilon {epsilon:.3f}"
                )
                ep_score = 0
                state = env.reset()
                if episode % config.SAVE_EVERY_EPISODES == 0:
                    save()

            if not headless:
                if not fast or steps % 10 == 0:
                    _draw(screen, font, env, show_sensors, show_route, placing, fast, episode,
                          ep_score, max_score, avg_score, epsilon, recent_goals, paused, user_paused)
                if not fast:
                    clock.tick(config.FPS)
    finally:
        save()
        if not headless:
            pygame.quit()

    return {"episodes": episode, "steps": steps, "max_score": max_score,
            "goals": goals, "no_route": no_route}


def _draw(screen, font, env, show_sensors, show_route, placing, fast, episode, ep_score,
          max_score, avg_score, epsilon, recent_goals, paused, user_paused):
    screen.fill(config.BG_COLOR, (0, 0, config.GAME_W, config.GAME_H))
    env.world.draw(screen)
    if show_route and env.route_ok:
        ui.draw_route(screen, env.route.path(env.car.x, env.car.y))
    ui.draw_markers(screen, env.world, placing)
    if show_sensors and not paused:
        ui.draw_sensors(screen, env.world, env.car)
    env.car.draw(screen, config.AI_CAR_COLOR)
    if placing:
        hint = f"Click the map to place the {'START' if placing == 'start' else 'FINISH'}  (Esc cancels)"
    else:
        hint = ui.HINT_PLACE
    lines = [
        f"Episode: {episode}   Score: {ep_score}   Max: {int(max_score)}   "
        f"Avg (last 20): {avg_score:.1f}",
        f"Reached finish: {sum(recent_goals)} of last {len(recent_goals)} episodes   "
        f"Epsilon: {epsilon:.3f}   Speed: {'FAST' if fast else 'normal'}",
        ui.HINT_DRAW,
        hint,
        ui.HINT_VIEW_AI,
    ]
    if paused:
        lines.append(NO_ROUTE_MESSAGE)
    ui.draw_panel(screen, font, lines)
    if user_paused:
        ui.draw_status(screen, pygame.font.Font(None, 40), font, "PAUSED",
                       "press Space to resume", config.TEXT_COLOR)
    throttle, brake, steer = env.last_controls
    ui.draw_steering_wheel(screen, steer)
    ui.draw_pedal(screen, ui.THROTTLE_RECT, throttle, config.THROTTLE_COLOR, "Gas", font)
    ui.draw_pedal(screen, ui.BRAKE_RECT, brake, config.BRAKE_COLOR, "Brake", font)
    pygame.display.flip()


def main():
    train(use_route="--no-route" not in sys.argv)


if __name__ == "__main__":
    main()
