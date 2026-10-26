# Self-driving car simulator

Pygame car simulator plus a PyTorch Double DQN that learns to drive from a start point to a finish point on a map the user draws. Runs on Windows in VS Code. Built from two tutorial videos, then extended (route compass, placeable start/finish, pause, eraser, no-route mode).

## Run
- Setup: `python -m venv .venv`, activate it, `pip install -r requirements.txt` (uses `pygame-ce`, `numpy`, `torch`, `pytest`).
- Human driving: `python car_hum.py` (add `--map NAME` to use `maps/NAME.json`)
- AI training: `python self_driving_car_ai.py` (add `--no-route` for the harder mode, `--cars N` for N cars at once; `--cars 1` is the original single-car training; `--map NAME` uses `maps/NAME.json`, all three combine)
- Tests: `python -m pytest` (tests set the SDL dummy video driver in `tests/conftest.py`; no window opens)

## Layout
- `car_sim/config.py` - EVERY tunable value. Add new settings here, never hard-code numbers elsewhere.
- `car_sim/car.py` - car physics. `car_sim/world.py` - walls, start/goal, mouse drawing/erasing, vectorized sensor rays, `version` counter that bumps when walls or goal change.
- `car_sim/route.py` - `RouteField` (grid Dijkstra from the finish, wall clearance + soft cost, rebuilt only when `world.version` changes) and `StraightField` (no-route mode).
- `car_sim/driving_env.py` - `DrivingEnv`: state, reward, episode end, reset, step. `compute_reward` is a pure function.
- `car_sim/vec_env.py` - `VecDrivingEnv`: N ghost cars (no car-to-car collisions) on one world with one shared route; a finished car resets itself and is reported in a `finished` list.
- `car_sim/dqn.py` - `QNetwork`, `ReplayBuffer`, `DoubleDQNAgent` (`act` for one state, `act_batch` for all cars).
- `car_sim/maps.py` - save/load maps as JSON (`{"version", "walls", "start", "goal"}`). Never raises for a missing or damaged file: returns a message and leaves the world untouched. `World.replace` swaps a map in and bumps `world.version`.
- `maps/` - saved maps, `maps/NAME.json`, created on the first Ctrl+S (relative to where you run the program, like `model.pt`). `maps/default.json` is loaded at startup if it exists; a missing default is silent, a missing `--map` file or a damaged file shows a panel message and starts empty.
- `car_sim/ui.py` - panel layout constants, status text, markers, route overlay, placement keys, `is_save_event` (Ctrl+S).
- `car_hum.py` (human mode) and `self_driving_car_ai.py` (`train()` loop and drawing).
- `docs/superpowers/specs/` - design specs. Read the spec for a feature before changing it. The latest: `2026-10-26-multi-car-design.md` (implemented; its default-setting measurements are not run yet).

## Key parameters (current values, all in `config.py`)
- Map 1400x800 with a 170 px panel. START (100,400), GOAL (1300,400), GOAL_RADIUS 30.
- State is 8 numbers: 5 sensors (-60..60 deg, /200 px), speed/140, route direction relative to heading (/pi), route distance/1600. With no route: direction/distance are straight-line to the finish, or 0 and 1 when there is no route.
- 5 steering actions `STEER_MAP = [-1, -0.5, 0, 0.5, 1]`; gas and brake are rule-based, not learned.
- Network 8 -> 64 -> 64 -> 5 (ReLU). Double DQN, gamma 0.99, Adam LR 3e-4, batch 64, replay 50,000, target update every 500 steps, epsilon 1.0 -> 0.05 with x0.9998 per step.
- Rewards: goal +20, crash -10, stall -5 (120 steps), timeout -5 (1800 steps), progress +0.1 per pixel of route distance closed (negative when moving away), slow -0.05, closeness up to -1, fast and close -2.
- Route: 10 px grid, hard clearance 24 px (the car crashes at 20), soft cost zone 56 px.
- Models save to `model.pt` (route mode) and `model_no_route.pt`.
- Multi-car: `NUM_CARS` 8 (unmeasured default), `UPDATES_PER_LOOP` 2, capped at the number of cars. Steps, epsilon decay and target updates count transitions (one per car per loop). `train(num_cars=1)` is the function default; `main()` uses `NUM_CARS` or `--cars`.

## Rules for changes
- Changing the state size, the number of actions or the network shape makes saved models unusable. Say so and tell the user to delete the `.pt` files and retrain.
- Do not use TDD shortcuts: write the failing test first, then the code. Keep all tests passing (214 at last count).
- Keep `pygame-ce`, not `pygame`. Keep a single-car path working when adding features.
- Prefer measuring to guessing for learning behavior; training results vary between checkpoints, so one run proves little. Do not claim an improvement without numbers.
- Run the tests before saying something works.

## Known behavior and decisions
- The route compass tells the car which way to go; the car still learns steering, braking comfort and avoiding walls. Progress reward nets to zero over loops, so dead ends are only weakly penalized (time and closeness costs).
- The UI calls the code's `ep_score` "Time alive" (simulated seconds = steps / `config.FPS`); "Longest" and "Avg time (last 20)" are `max_score` and `avg_score` in the same units.
- Ctrl+S saves the map. Human mode brakes on `S`, so `read_keys` ignores driving keys while Ctrl is held. Map names are 1-20 letters, digits, `_` or `-` (longer names would overflow the panel message).
- `tests/conftest.py` points `config.MAPS_DIR` at a temp folder for every test, so a saved `maps/default.json` never changes test results. `train(world=...)` does not load a map file; only a `train()` that makes its own world does.
- Timeout is treated as a terminal state for learning.
- Lowering LR from 1e-3 to 3e-4 made the saved policy far more stable (84/100 vs 35/100 greedy successes in one test).
- The user is a CS graduate learning this project. Explain the why briefly when you change something, and prefer short, direct steps.
