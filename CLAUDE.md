# Self-driving car simulator - project guide for Claude Code

Pygame car simulator plus a PyTorch Double DQN that learns to drive from a start point to a finish point on a map the user draws. Runs on Windows in VS Code. Built from two tutorial videos, then extended (route compass, placeable start/finish, pause, eraser, no-route mode).

## Run
- Setup: `python -m venv .venv`, activate it, `pip install -r requirements.txt` (uses `pygame-ce`, `numpy`, `torch`, `pytest`).
- Human driving: `python car_hum.py`
- AI training: `python self_driving_car_ai.py` (add `--no-route` for the harder mode)
- Tests: `python -m pytest` (tests set the SDL dummy video driver in `tests/conftest.py`; no window opens)

## Layout
- `car_sim/config.py` - EVERY tunable value. Add new settings here, never hard-code numbers elsewhere.
- `car_sim/car.py` - car physics. `car_sim/world.py` - walls, start/goal, mouse drawing/erasing, vectorized sensor rays, `version` counter that bumps when walls or goal change.
- `car_sim/route.py` - `RouteField` (grid Dijkstra from the finish, wall clearance + soft cost, rebuilt only when `world.version` changes) and `StraightField` (no-route mode).
- `car_sim/driving_env.py` - `DrivingEnv`: state, reward, episode end, reset, step. `compute_reward` is a pure function.
- `car_sim/dqn.py` - `QNetwork`, `ReplayBuffer`, `DoubleDQNAgent`.
- `car_sim/ui.py` - panel layout constants, status text, markers, route overlay, placement keys.
- `car_hum.py` (human mode) and `self_driving_car_ai.py` (`train()` loop and drawing).
- `docs/superpowers/specs/` - design specs. Read the spec for a feature before changing it. The current one: `2026-10-26-multi-car-design.md` (not implemented yet).

## Key parameters (current values, all in `config.py`)
- Map 1400x800 with a 170 px panel. START (100,400), GOAL (1300,400), GOAL_RADIUS 30.
- State is 8 numbers: 5 sensors (-60..60 deg, /200 px), speed/120, route direction relative to heading (/pi), route distance/1600. With no route: direction/distance are straight-line to the finish, or 0 and 1 when there is no route.
- 5 steering actions `STEER_MAP = [-1, -0.5, 0, 0.5, 1]`; gas and brake are rule-based, not learned.
- Network 8 -> 64 -> 64 -> 5 (ReLU). Double DQN, gamma 0.99, Adam LR 3e-4, batch 64, replay 50,000, target update every 500 steps, epsilon 1.0 -> 0.05 with x0.9998 per step.
- Rewards: goal +20, crash -10, stall -5 (120 steps), timeout -5 (3000 steps), progress +0.1 per pixel of route distance closed (negative when moving away), slow -0.05, closeness up to -1, fast and close -2.
- Route: 10 px grid, hard clearance 24 px (the car crashes at 20), soft cost zone 56 px.
- Models save to `model.pt` (route mode) and `model_no_route.pt`.

## Rules for changes
- Changing the state size, the number of actions or the network shape makes saved models unusable. Say so and tell the user to delete the `.pt` files and retrain.
- Do not use TDD shortcuts: write the failing test first, then the code. Keep all tests passing (130 at last count).
- Keep `pygame-ce`, not `pygame`. Keep a single-car path working when adding features.
- Prefer measuring to guessing for learning behavior; training results vary between checkpoints, so one run proves little. Do not claim an improvement without numbers.
- Run the tests before saying something works.

## Known behavior and decisions
- The route compass tells the car which way to go; the car still learns steering, braking comfort and avoiding walls. Progress reward nets to zero over loops, so dead ends are only weakly penalized (time and closeness costs).
- Timeout is treated as a terminal state for learning.
- Lowering LR from 1e-3 to 3e-4 made the saved policy far more stable (84/100 vs 35/100 greedy successes in one test).
- The user is a CS graduate learning this project. Explain the why briefly when you change something, and prefer short, direct steps.
