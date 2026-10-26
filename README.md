# Self-Driving Car Simulator (Pygame + PyTorch Double DQN)

A 2D car simulator you can drive yourself, and the same simulator driven by a
Double DQN agent that learns to drive from a **start point to a finish point**
around walls you draw with the mouse while it runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs `pygame-ce`, a drop-in fork of pygame that ships
ready-made packages for new Python versions. If you prefer the original, the
`pygame` package works on older Python versions but may fail to build on very
new ones (for example Python 3.14); `pip install pygame-ce` fixes that.

## Run

```bash
python car_hum.py                # you drive
python self_driving_car_ai.py    # the AI drives and trains live (8 cars at once by default)
python self_driving_car_ai.py --no-route   # harder mode, see below
python self_driving_car_ai.py --cars 4     # train with 4 cars at once (--cars 1: a single car)
python car_hum.py --map maze               # use maps/maze.json (works in both programs)
```

The window is 1400 x 800 plus a control panel. To change the map size, edit
`GAME_W` and `GAME_H` in `car_sim/config.py` (move `START_X/Y` and `GOAL_X/Y`
with them).

## Controls

| Input | Human mode (`car_hum.py`) | AI mode (`self_driving_car_ai.py`) |
|---|---|---|
| Mouse: left-drag | draw walls | draw walls |
| Mouse: right-drag | erase walls near the cursor | erase walls near the cursor |
| `Space` | pause / resume | pause / resume |
| `1`, then click the map | place the **start** | place the **start** |
| `2`, then click the map | place the **finish** | place the **finish** |
| `Esc` | cancel placing | cancel placing |
| `P` | show/hide the shortest route | show/hide the shortest route |
| `V` | show/hide sensors | show/hide sensors |
| `C` | clear walls (start and finish stay) | clear walls (start and finish stay) |
| `Ctrl+S` | save the map to `maps/default.json` (or `maps/NAME.json` with `--map NAME`) | same |
| `Up`/`W`, `Down`/`S` | throttle, brake | (AI controls) |
| `Left`/`A`, `Right`/`D` | steer | (AI controls) |
| `R` | restart after a crash or a win | (restarts automatically) |
| `F` | - | fast mode: runs faster than 60 FPS |

The green dot is the start, the red circle is the finish. Reaching the circle
wins. Placing a point does not draw a wall.

In the AI window every car is drawn. The first one is the **focus car**: it is solid, it is the only one that shows sensors (`V`) and the route (`P`), and the gas, brake and steering display and the "Time alive" figure are its own. The others are see-through. The panel shows `Cars: N`.

## Saving and loading maps

`Ctrl+S` writes the walls, start and finish to a JSON file in the `maps/` folder
(created the first time you save) and shows "Saved maps/default.json" in the
panel. `maps/default.json` is loaded automatically at startup if it exists. Use
`--map NAME` with either program to load and save `maps/NAME.json` instead; it
combines with `--cars` and `--no-route`. Names are up to 20 letters, digits, `_`
or `-`. The folder is relative to where you run the program, like `model.pt`.

A file is `{"version": 1, "walls": [[x1, y1, x2, y2], ...], "start": [x, y], "goal": [x, y]}`.
If a map file you asked for with `--map` is missing, or any map file is damaged
or from a newer version, the program shows a message in the panel and starts with
the empty map instead of crashing. A missing `default.json` is not an error.
The saved model is separate (`model.pt`) and is not tied to a map.

## How it works

- **Physics:** top speed is 140 px/s (`MAX_SPEED`), reached at 140 px/s² of acceleration (`MAX_ACCEL`). Braking is 2.5 times stronger than accelerating, coasting loses 60 px/s² to friction, and the car can turn at up to 120 degrees per second. Steering is the same at any speed (`STEER_SPEED_SCALE = 0`) so the car can turn tightly. The car is 40 x 20 px and crashes when its center gets within 20 px of a wall. All of these are in `config.py`.
- **Route compass:** the program flood-fills a coarse grid (10 px cells, walls thickened by 24 px, just past the car's crash distance of 20 px) outward from the finish. Every cell then knows the length of the shortest route around the walls and which neighbor is the next step. Gaps narrower than about 50 px count as walls for routing. The route also pays extra for running close to a wall (inside about 56 px), so it follows the middle of a corridor and takes corners wide instead of hugging the inside, which is where a car clips a wall.
- **State (8 numbers):** five sensor distances (at -60, -30, 0, 30, 60 degrees), speed, the direction of the route relative to where the car points (0 = straight ahead), and how much route is left. All scaled to 0-1 (the direction to -1..1).
- **Actions (5):** steer from full left to full right. Throttle and braking follow simple rules (brake harder the closer an obstacle is ahead).
- **Rewards:** positive for getting closer to the finish *along the route* (negative for moving away), a big bonus for reaching it, penalties for crawling and for getting close to walls (side sensors count too, bigger at speed), a large negative for crashing, a smaller one for running out of time.
- **Double DQN:** the online network picks the next action and the target network scores it, which keeps value estimates from running too optimistic. A replay buffer of past experiences is sampled in batches to train.
- **Several cars at once:** training runs `NUM_CARS` cars (8 by default, `--cars N` to change) on the same map. They are ghosts that pass through each other. All of them use the one network and add to the one replay buffer, and one forward pass picks every car's action. When a car crashes, stalls, times out or reaches the finish, only that car restarts; the others keep driving, and the route is built once and shared. This is meant to collect experience faster in wall-clock time. `--cars 1` is the original single-car training, and the saved model works with any number of cars.
- **What counts as a step:** a step is one car's transition, so with N cars one loop is N steps. Epsilon decay and the target-network updates count steps, so the exploration schedule covers the same amount of experience whatever N is. Each loop does `UPDATES_PER_LOOP` (2) training updates, never more than the number of cars, so one car gets one update per step as before.
- **What the AI does and does not know:** it has no memory of the map. The route compass supplies the plan (so a dead-end pocket does not trap it), and the network learns the driving: steering, keeping off walls, using momentum.
- **`--no-route` mode:** the car is told only the straight-line direction and distance to the finish, not the route around walls, so it has to learn to get around them by crashing. It is much harder and mostly works on open maps. It saves to its own file, `model_no_route.pt`, so the two modes never overwrite each other.
- **No route?** If walls completely block the finish, the AI window pauses training and says so. Move a point or open a gap and it resumes by itself. Headless runs stop immediately instead.

## Training tips

- Early on the car looks random and crashes a lot. That is exploration (epsilon starts at 1.0 and decays to 0.05 over about 15,000 steps, which is about 1,900 loops with 8 cars).
- Draw walls early, then change them as it learns. Moving the finish or drawing a wall mid-run is fine; the route updates on the fly.
- By default every episode starts at your start point, pointing roughly along the route (within 30 degrees). Set `RANDOM_SPAWN = True` in `config.py` to start from random free spots instead.
- Press `F` to train faster. At normal speed the simulation runs 60 loops per second (60 steps per second per car); fast mode removes that cap and draws only every 10th loop.
- **Is 8 cars better than 1? Not measured yet.** The default of 8 cars and 2 updates per loop are untested guesses. Cars started together on similar paths give similar experience, the 50,000-entry replay buffer fills N times faster so older experience is dropped sooner (raise `MEMORY_SIZE` if that matters), and the speed-up is less than N because every car still has to be simulated. The plan for deciding the defaults is in `docs/superpowers/specs/2026-10-26-multi-car-design.md`. Until then, compare `--cars 1` and `--cars 8` on your own map before trusting either.
- The model saves to `model.pt` every 5 episodes and when you close the window, and loads automatically next time. Delete `model.pt` to start over. A `model.pt` from the earlier 6-number version is not compatible; the program says so once and starts fresh.

## Measured results

Headless test on a 1400x800 map with a U-shaped dead-end pocket in the straight line between start and finish (start at the left, finish at the right) plus two more walls forming an S-bend. Start to finish is about 1,370 px along the shortest route. These are single runs of 150,000 training steps with one car (about 42 minutes at normal speed, roughly 5-7 minutes in fast mode), so treat them as indicative, not guaranteed. They were taken before multi-car training existed, and they were not repeated after the car speed and episode time limit in `config.py` were changed to their current values:

- **During training**, in the last 50 episodes the car reached the finish in 42 (84%), crashed in 4 and stalled in 4, with an average episode length of about 880 steps.
- **The saved model is a snapshot, and snapshots vary.** The final saved model reached the finish in 4 of 20 greedy runs and 10 of 20 runs with 5% random moves. Checking a separate 100,000-step run every 10,000 steps (10 greedy runs each), it reached the finish in 84 of 100 runs overall, but some individual checkpoints were much worse than others. This wobble is typical of DQN. If a run looks worse after you stop, let it train a bit longer.
- **Learning rate matters:** `LR` was lowered from 1e-3 to 3e-4 after the same check at 1e-3 reached the finish in only 35 of 100 checkpoint runs.
- Not measured: how well it does on maps with many more walls, or visually (these runs had no window).

## Tuning

Every number lives in `car_sim/config.py`: window size, start/finish, car physics, sensors, reward weights, learning settings, and the multi-car settings (`NUM_CARS`, `UPDATES_PER_LOOP`, `GHOST_CAR_ALPHA`). Changing `STATE_DIM`, `ACTION_DIM` or `HIDDEN_SIZE` makes saved `.pt` models unusable; delete them and retrain.

## Layout

```
car_sim/
  config.py        all constants
  car.py           car physics
  world.py         walls, start/finish points, sensor rays, collisions
  maps.py          saving and loading maps as JSON
  route.py         shortest route around the walls to the finish
  ui.py            panel, steering wheel, pedals, markers, route overlay, placement keys
  driving_env.py   state, controller rules, rewards, episode end conditions (one car)
  vec_env.py       several cars on one map sharing one route (multi-car training)
  dqn.py           network, replay buffer, Double DQN training, batched action choice
maps/                      saved maps, maps/NAME.json (created by Ctrl+S)
docs/superpowers/          design specs (specs/) and implementation plans (plans/)
car_hum.py                 human-driven simulator
self_driving_car_ai.py     AI-driven simulator with live training
tests/                     headless tests (run: pytest)
CLAUDE.md                  project guide for Claude Code: commands, layout, rules for changes
```

## Tests

```bash
python -m pytest
```

149 automated tests cover the car physics, sensors, route finding, rewards,
the Double DQN update, multi-car training, map saving and the UI. They run
headless (no window, no GPU needed), so they work on any computer or in CI.
The project was built test first: each feature starts as a failing test.
