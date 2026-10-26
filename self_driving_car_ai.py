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

`--cars N` drives N cars at once (default config.NUM_CARS); they share one network and one
replay buffer and pass through each other. `--cars 1` is the original single-car training.
"""
import argparse
import os
import sys
from collections import deque

import pygame

from car_sim import config, ui
from car_sim.dqn import DoubleDQNAgent
from car_sim.vec_env import VecDrivingEnv
from car_sim.world import World

NO_ROUTE_MESSAGE = "No route from start to finish: move a point or open a gap"


def train(headless=False, max_steps=None, model_path=None, world=None, use_route=True,
          num_cars=1):
    """Run the training loop.

    Returns {"episodes", "steps", "max_score", "goals", "no_route"}. Steps are transitions
    (one per car per loop); max_steps counts them in this run only and is reached in whole
    loops, so it can be overshot by up to num_cars - 1. episodes and max_score carry over
    from a loaded model. headless=True skips the window; if the start has no route it then
    returns at once (no_route True), because nobody is there to fix the map.
    """
    if model_path is None:
        model_path = config.MODEL_PATH if use_route else config.MODEL_PATH_NO_ROUTE
    vec = VecDrivingEnv(world if world is not None else World(), num_cars, use_route)
    agent = DoubleDQNAgent()
    # More gradient steps than new transitions would just replay old data harder.
    updates_per_loop = min(num_cars, config.UPDATES_PER_LOOP)
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
    steps = 0
    loops = 0
    avg_score = 0.0
    no_route = False
    paused = False        # no route from start to finish
    user_paused = False   # the player pressed Space
    states = vec.reset()
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
                    before_start = vec.world.start
                    new_mode = ui.handle_placement_event(event, vec.world, placing)
                    if event.type == pygame.MOUSEBUTTONDOWN and placing and new_mode is None:
                        no_draw_until_release = True  # that click placed a point, not a wall
                    placing = new_mode
                    if vec.world.start != before_start:
                        states = vec.reset()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_v:
                            show_sensors = not show_sensors
                        elif event.key == pygame.K_p:
                            show_route = not show_route
                        elif event.key == pygame.K_SPACE:
                            user_paused = not user_paused
                        elif event.key == pygame.K_c:
                            vec.world.clear()
                        elif event.key == pygame.K_f:
                            fast = not fast
                held, _, right = pygame.mouse.get_pressed()[:3]
                if not held:
                    no_draw_until_release = False
                vec.world.handle_mouse(
                    pygame.mouse.get_pos(), held and placing is None and not no_draw_until_release
                )
                vec.world.handle_erase(pygame.mouse.get_pos(), right and placing is None)
                if not running:
                    break
                if user_paused:
                    _draw(screen, font, vec, show_sensors, show_route, placing, fast, episode,
                          max_score, avg_score, epsilon, recent_goals, paused, True)
                    clock.tick(config.FPS)
                    continue

            # Nothing to learn from while the finish cannot be reached.
            if not vec.check_route():
                if headless:
                    print(NO_ROUTE_MESSAGE)
                    no_route = True
                    break
                paused = True
                _draw(screen, font, vec, show_sensors, show_route, placing, fast, episode,
                      max_score, avg_score, epsilon, recent_goals, paused, user_paused)
                clock.tick(config.FPS)
                continue
            if paused:  # the map was fixed: start clean episodes
                paused = False
                states = vec.reset()

            actions = agent.act_batch(states, epsilon)
            next_states, rewards, dones, finished = vec.step(actions)
            for transition in zip(states, actions, rewards, next_states, dones):
                agent.remember(*transition)
            for _ in range(updates_per_loop):
                agent.train_step()

            states = list(vec.states)
            loops += 1
            # Epsilon and the target network count transitions, so the schedule means the
            # same number of experiences whatever the number of cars.
            before = steps
            steps += num_cars
            epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY ** num_cars)
            if steps // config.TARGET_UPDATE_STEPS != before // config.TARGET_UPDATE_STEPS:
                agent.update_target()

            for ended in finished:
                episode += 1
                reached = ended["outcome"] == "goal"
                goals += int(reached)
                recent_goals.append(reached)
                ep_score = ended["score"]
                max_score = max(max_score, ep_score)
                recent.append(ep_score)
                avg_score = sum(recent) / len(recent)
                print(
                    f"Episode {episode} | {ended['outcome']} | time alive {ep_score / config.FPS:.1f}s "
                    f"| longest {max_score / config.FPS:.1f}s "
                    f"| avg time (last 20) {avg_score / config.FPS:.1f}s | goals {goals} | epsilon {epsilon:.3f}"
                )
                if episode % config.SAVE_EVERY_EPISODES == 0:
                    save()

            if not headless:
                if not fast or loops % 10 == 0:
                    _draw(screen, font, vec, show_sensors, show_route, placing, fast, episode,
                          max_score, avg_score, epsilon, recent_goals, paused, user_paused)
                if not fast:
                    clock.tick(config.FPS)
    finally:
        save()
        if not headless:
            pygame.quit()

    return {"episodes": episode, "steps": steps, "max_score": max_score,
            "goals": goals, "no_route": no_route}


def _draw(screen, font, vec, show_sensors, show_route, placing, fast, episode,
          max_score, avg_score, epsilon, recent_goals, paused, user_paused):
    # The first car is the focus car: route, sensors, controls and time alive are its own.
    env = vec.envs[0]
    ep_score = env.steps
    screen.fill(config.BG_COLOR, (0, 0, config.GAME_W, config.GAME_H))
    vec.world.draw(screen)
    if show_route and vec.route_ok:
        ui.draw_route(screen, vec.route.path(env.car.x, env.car.y))
    ui.draw_markers(screen, vec.world, placing)
    for ghost in vec.cars[1:]:
        ghost.draw(screen, config.AI_CAR_COLOR, alpha=config.GHOST_CAR_ALPHA)
    if show_sensors and not paused:
        ui.draw_sensors(screen, vec.world, env.car)
    env.car.draw(screen, config.AI_CAR_COLOR)
    if placing:
        hint = f"Click the map to place the {'START' if placing == 'start' else 'FINISH'}  (Esc cancels)"
    else:
        hint = ui.HINT_PLACE
    lines = [
        f"Episode: {episode}   Time alive: {ep_score / config.FPS:.1f}s   "
        f"Longest: {max_score / config.FPS:.1f}s   "
        f"Avg time (last 20): {avg_score / config.FPS:.1f}s",
        f"Reached finish: {sum(recent_goals)} of last {len(recent_goals)} episodes   "
        f"Epsilon: {epsilon:.3f}   Speed: {'FAST' if fast else 'normal'}   Cars: {len(vec.envs)}",
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


def parse_args(argv):
    """Command line -> (use_route, num_cars)."""
    def car_count(text):
        n = int(text)
        if n < 1:
            raise argparse.ArgumentTypeError("needs at least 1 car")
        return n

    parser = argparse.ArgumentParser(description="Watch a Double DQN agent learn to drive.")
    parser.add_argument("--no-route", action="store_true",
                        help="harder mode: only the straight line to the finish is known")
    parser.add_argument("--cars", type=car_count, default=config.NUM_CARS,
                        help=f"cars training at once (default {config.NUM_CARS})")
    args = parser.parse_args(argv)
    return not args.no_route, args.cars


def main():
    use_route, num_cars = parse_args(sys.argv[1:])
    train(use_route=use_route, num_cars=num_cars)


if __name__ == "__main__":
    main()
