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
python self_driving_car_ai.py    # the AI drives and trains live
python self_driving_car_ai.py --no-route   # harder mode, see below
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
| `Up`/`W`, `Down`/`S` | throttle, brake | (AI controls) |
| `Left`/`A`, `Right`/`D` | steer | (AI controls) |
| `R` | restart after a crash or a win | (restarts automatically) |
| `F` | - | fast mode: runs faster than 60 FPS |

The green dot is the start, the red circle is the finish. Reaching the circle
wins. Placing a point does not draw a wall.

## How it works

- **Physics:** top speed is 120 px/s, and steering is the same at any speed so the car can turn tightly. Both are in `config.py`.
- **Route compass:** the program flood-fills a coarse grid (10 px cells, walls thickened by 24 px, just past the car's crash distance of 20 px) outward from the finish. Every cell then knows the length of the shortest route around the walls and which neighbor is the next step. Gaps narrower than about 50 px count as walls for routing. The route also pays extra for running close to a wall (inside about 56 px), so it follows the middle of a corridor and takes corners wide instead of hugging the inside, which is where a car clips a wall.
- **State (8 numbers):** five sensor distances (at -60, -30, 0, 30, 60 degrees), speed, the direction of the route relative to where the car points (0 = straight ahead), and how much route is left. All scaled to 0-1 (the direction to -1..1).
- **Actions (5):** steer from full left to full right. Throttle and braking follow simple rules (brake harder the closer an obstacle is ahead).
- **Rewards:** positive for getting closer to the finish *along the route* (negative for moving away), a big bonus for reaching it, penalties for crawling and for getting close to walls (side sensors count too, bigger at speed), a large negative for crashing, a smaller one for running out of time.
- **Double DQN:** the online network picks the next action and the target network scores it, which keeps value estimates from running too optimistic. A replay buffer of past experiences is sampled in batches to train.
- **What the AI does and does not know:** it has no memory of the map. The route compass supplies the plan (so a dead-end pocket does not trap it), and the network learns the driving: steering, keeping off walls, using momentum.
- **`--no-route` mode:** the car is told only the straight-line direction and distance to the finish, not the route around walls, so it has to learn to get around them by crashing. It is much harder and mostly works on open maps. It saves to its own file, `model_no_route.pt`, so the two modes never overwrite each other.
- **No route?** If walls completely block the finish, the AI window pauses training and says so. Move a point or open a gap and it resumes by itself. Headless runs stop immediately instead.

## Training tips

- Early on the car looks random and crashes a lot. That is exploration (epsilon starts at 1.0 and decays to 0.05 over about 15,000 steps).
- Draw walls early, then change them as it learns. Moving the finish or drawing a wall mid-run is fine; the route updates on the fly.
- By default every episode starts at your start point, pointing roughly along the route (within 30 degrees). Set `RANDOM_SPAWN = True` in `config.py` to start from random free spots instead.
- Press `F` to train faster. At normal speed the simulation runs at 60 steps per second; fast mode removes that cap.
- The model saves to `model.pt` every 5 episodes and when you close the window, and loads automatically next time. Delete `model.pt` to start over. A `model.pt` from the earlier 6-number version is not compatible; the program says so once and starts fresh.

## Measured results

Headless test on a 1400x800 map with a U-shaped dead-end pocket in the straight line between start and finish (start at the left, finish at the right) plus two more walls forming an S-bend. Start to finish is about 1,370 px along the shortest route. These are single runs of 150,000 training steps (about 42 minutes at normal speed, roughly 5-7 minutes in fast mode), so treat them as indicative, not guaranteed:

- **During training**, in the last 50 episodes the car reached the finish in 42 (84%), crashed in 4 and stalled in 4, with an average episode length of about 880 steps.
- **The saved model is a snapshot, and snapshots vary.** The final saved model reached the finish in 4 of 20 greedy runs and 10 of 20 runs with 5% random moves. Checking a separate 100,000-step run every 10,000 steps (10 greedy runs each), it reached the finish in 84 of 100 runs overall, but some individual checkpoints were much worse than others. This wobble is typical of DQN. If a run looks worse after you stop, let it train a bit longer.
- **Learning rate matters:** `LR` was lowered from 1e-3 to 3e-4 after the same check at 1e-3 reached the finish in only 35 of 100 checkpoint runs.
- Not measured: how well it does on maps with many more walls, or visually (these runs had no window).

## Tuning

Every number lives in `car_sim/config.py`: window size, start/finish, car physics, sensors, reward weights, and learning settings.

## Layout

```
car_sim/
  config.py        all constants
  car.py           car physics
  world.py         walls, start/finish points, sensor rays, collisions
  route.py         shortest route around the walls to the finish
  ui.py            panel, steering wheel, pedals, markers, route overlay, placement keys
  driving_env.py   state, controller rules, rewards, episode end conditions
  dqn.py           network, replay buffer, Double DQN training
car_hum.py                 human-driven simulator
self_driving_car_ai.py     AI-driven simulator with live training
tests/                     headless tests (run: pytest)
```

## Tests

```bash
pytest
```
