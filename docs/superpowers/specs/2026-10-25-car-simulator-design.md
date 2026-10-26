# Car Simulator + Double DQN Self-Driving Car: Design

Date: 2026-10-25
Status: awaiting user review

## Goal

Recreate the two tutorial projects as one working codebase the user can run locally and learn from:

1. A Pygame car simulator the user drives by hand (video 1).
2. The same simulator driven by a Double DQN agent trained live (video 2).

Success means: both programs run on the user's machine, the human and AI cars follow identical physics, and the agent visibly learns to avoid drawn obstacles.

## Decisions (agreed with user)

- **Framework:** PyTorch (not TensorFlow/Keras as in the video).
- **Structure:** a small shared package with two thin entry scripts (not two standalone files).
- **Obstacles:** held in a `World` object instead of a global list.
- **Addition beyond the videos:** save/load of the trained model.
- **Source material:** the two video transcripts in the Car Simulator project. Fixes shown in the videos are applied from the start: negative crash reward, integer pixel coordinates in `pygame.draw` calls, score only increases when the car actually moves.
- Values the videos do not state (network sizes, some constants, reward weights) get sensible defaults, all in `config.py`.

## Layout

```
car_sim/
  __init__.py
  config.py        constants: window/panel size, colors, car physics, sensors, RL hyperparameters, reward weights
  car.py           Car: x, y, angle, speed, angular velocity; step(dt, throttle, brake, steer) -> distance moved; draw(surface); reset()
  world.py         World: obstacles (pairs of points), point_segment_distance(), cast_ray(), collided(car), mouse drawing, clear(), draw_obstacles()
  ui.py            panel drawing, steering wheel, throttle/brake pedal bars, text
  driving_env.py   DrivingEnv: wraps Car + World for the AI: reset() -> state, step(action) -> (state, reward, done); get_state(); compute_reward()
  dqn.py           QNetwork, ReplayBuffer, DoubleDQNAgent (act, remember, train_step, update_target, save, load)
car_hum.py                 human-driven simulator
self_driving_car_ai.py     AI-driven simulator with live training
tests/                     headless pytest tests
requirements.txt
README.md
```

## Component details

### Simulator core (`car.py`, `world.py`)

- Window: 800x600 game area plus an extra bottom panel. Pygame coordinates (y down). Angle drawn as `-degrees(angle)`.
- Car physics per frame, all scaled by delta time:
  - acceleration = throttle * MAX_ACCEL; braking = brake * MAX_ACCEL * BRAKE_FORCE; speed += (accel - braking) * dt.
  - If throttle is 0, friction reduces speed. Speed clamped to [0, MAX_SPEED].
  - Steering: target angular velocity = steer * MAX_STEER_RATE (radians); current angular velocity moves toward it by STEER_SMOOTHING; angle += angular velocity * dt.
  - Position advances by (cos(angle), sin(angle)) * speed * dt. `step` returns distance moved.
- Sensors: 5 rays at -60, -30, 0, 30, 60 degrees relative to heading, max distance MAX_SENSOR_DISTANCE, marched in SENSOR_STEP increments. A ray stops at the game-area boundary or when within 2 px of an obstacle segment.
- `point_segment_distance(p, a, b)`: project p onto ab, clamp t to [0, 1], return distance to the projected point; if a == b return distance to a.
- Collision: car center within half its width/height of the boundary, or closer than the car size to any obstacle segment.
- Obstacle drawing: left mouse drag in the game area adds connected segments; releasing resets the previous point. `C` clears obstacles.
- Rendering: car surface uses per-pixel alpha so rotation blends cleanly.

### Human mode (`car_hum.py`)

- Keys: Up/W throttle, Down/S brake, Left/A and Right/D steer, V toggle sensors, C clear obstacles, R restart after a crash.
- Panel shows score, max score, controls, a game-over message, the steering wheel, and throttle/brake bars.
- Score increases only when distance moved in a frame exceeds a small threshold. Crash sets game over and freezes updates.

### AI environment (`driving_env.py`)

- **State (6 floats, numpy array):** five sensor distances divided by MAX_SENSOR_DISTANCE, plus speed divided by MAX_SPEED.
- **Actions (5):** index maps through STEER_MAP to [-1, -0.5, 0, 0.5, 1].
- **Controller rules around the chosen steering** (as in the video): steering is scaled down as speed rises; brake rises as the front sensor reading falls and is clipped to [0, 1]; above a small brake threshold, throttle is cut and a brake-time counter increments; otherwise throttle is applied and reduced slightly during sharp steering.
- **Reward:**
  - positive reward for forward progress when not braking hard;
  - small penalty when speed is very low;
  - penalty that grows as the front sensor reading shrinks;
  - strong penalty when fast and the front obstacle is close;
  - large negative reward on collision;
  - negative stall penalty when the episode ends from stalling.
- **Episode end:** collision, or the car moving less than STALL_DISTANCE_EPS per step for STALL_LIMIT consecutive steps.

### Learning (`dqn.py`)

- `QNetwork`: 6 -> 64 -> 64 -> 5, ReLU, linear output (one Q value per action).
- `ReplayBuffer`: deque with maxlen MEMORY_SIZE (50,000); samples random batches and returns stacked tensors.
- `DoubleDQNAgent`:
  - epsilon-greedy `act` (epsilon decays from EPS_START 1.0 to EPS_END 0.05);
  - `train_step`: once memory holds at least BATCH_SIZE experiences, sample a batch, pick next actions with the online network, evaluate them with the target network, build Bellman targets `r + GAMMA * Q_target(s', argmax Q_online(s'))` (just `r` when done), and minimize MSE with Adam; vectorized, no per-sample loop;
  - `update_target` copies online weights to the target every TARGET_UPDATE_STEPS steps;
  - `save(path)` / `load(path)` store and restore network weights (and epsilon and episode count so a resumed run continues sensibly).
- Defaults: GAMMA 0.99, LR 1e-3, BATCH_SIZE 64, MEMORY_SIZE 50,000, TARGET_UPDATE_STEPS 500.

### AI mode (`self_driving_car_ai.py`)

- Same window, world, and obstacle drawing as human mode, with an aqua-blue car.
- Loop: get state, agent chooses action, env steps, remember, train, update epsilon and target network. On episode end, record the score, update max score and the 20-episode average, print the stats, reset.
- Panel shows episode, current score, max score, 20-episode average, and epsilon.
- Loads a saved model on start if one exists; saves periodically and on exit.

## Error handling

Minimal. A missing model file starts fresh training. A model file that cannot be loaded prints a clear message and starts fresh instead of crashing. Pygame and torch are assumed installed per `requirements.txt`.

## Testing (headless, pytest, `SDL_VIDEODRIVER=dummy`)

- Physics: acceleration, braking stronger than acceleration, friction, speed clamping, smooth steering.
- Geometry: `point_segment_distance` for projection inside the segment, before A, past B, and zero-length segments.
- Sensors and collisions: rays stop at walls and drawn obstacles; collision at the screen edge and at an obstacle.
- Reward and episode end: crash gives a negative reward; stall ends the episode.
- DQN: network output shape; one training step changes weights; target copy matches online; save/load round trip.
- End-to-end: a short headless training run of a few thousand steps completes without errors. This checks that it runs, not that the car learns well.

## Out of scope

Pre-built tracks, multi-agent racing, GPU-specific tuning, a hyperparameter search, TensorFlow support, and packaging for distribution.

## Known limits

The Pygame window cannot be opened in the build environment, so the first visual run happens on the user's machine. Learning quality depends on real training time; the video reports roughly 10 minutes to clear obstacles and hours to become strong.

## Revisions after verification (2026-10-25)

A headless training check showed the first defaults could not learn: at the original `MAX_SPEED = 200` with steering scaled down at speed, the turning radius was about 190 px (nearly the sensor range), the reward only read the front sensor, and the fixed spawn let the agent replay one doomed trajectory. Changes, all in `config.py`:

- `MAX_SPEED` 200 -> 120, and `STEER_SPEED_SCALE = 0.0` (steering no longer reduced at speed).
- The proximity penalty uses `proximity(state)`: the front sensor, or a side sensor plus `PROXIMITY_SIDE_MARGIN = 0.25`, whichever is closer.
- `RANDOM_SPAWN = True`: AI episodes start at a random free spot and heading (falls back to `START_X, START_Y`). Human mode keeps the fixed start.
- Added during review: segments are added only after the mouse moves `MOUSE_MIN_SEGMENT = 3` px (a click without movement leaves one dot on release); sensors are cast once per step; the sensor cache is invalidated on `add_segment` and `clear`; an `F` key toggles fast training.
