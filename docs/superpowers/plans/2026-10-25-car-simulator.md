# Car Simulator + Double DQN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Pygame car simulator the user can drive by hand, and the same simulator driven by a PyTorch Double DQN agent that trains live.

**Architecture:** A shared `car_sim` package (config, car physics, world with obstacles/sensors/collisions, UI, an AI-facing environment wrapper, and the DQN) with two thin entry scripts. The human and the AI use the same `Car` and `World`.

**Tech Stack:** Python 3.10+, pygame, numpy, torch (CPU), pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-car-simulator-design.md`

## Global Constraints

- Window: 800x600 game area plus a bottom panel (`PANEL_H = 150`). Pygame coordinates, y down; car drawn rotated by `-degrees(angle)`.
- Sensors: 5 rays at -60, -30, 0, 30, 60 degrees relative to heading; ray stops at the game-area boundary or within 2 px of an obstacle segment.
- State is 6 floats: 5 sensor distances / `MAX_SENSOR_DISTANCE`, then speed / `MAX_SPEED`. Actions: 5, `STEER_MAP = [-1, -0.5, 0, 0.5, 1]`.
- Network 6 -> 64 -> 64 -> 5 with ReLU. `GAMMA 0.99`, `LR 1e-3`, `BATCH_SIZE 64`, `MEMORY_SIZE 50000`, `TARGET_UPDATE_STEPS 500`, epsilon 1.0 -> 0.05.
- Crash reward is negative. Pygame draw calls get integer pixel coordinates. Human score only increases when the car actually moves.
- Every tunable value lives in `car_sim/config.py`. All tests run headless (`SDL_VIDEODRIVER=dummy`).

## Review Focus

- Clicking without dragging creates a zero-length obstacle segment: no crash, behaves as a point.
- A huge frame time (window drag, lag spike) must not teleport the car: `dt` is clamped to `MAX_DT = 0.05`.
- A missing or corrupt model file: training starts fresh and prints one clear message.
- An obstacle drawn on top of the car's spawn point: `reset()` still works and the episode ends on the first step, with no infinite reset loop.
- Closing the window mid-training saves the model.

---

### Task 1: Scaffold and config

**Files:**
- Create: `requirements.txt`, `car_sim/__init__.py`, `car_sim/config.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `car_sim.config` constants, all upper-case: `GAME_W=800`, `GAME_H=600`, `PANEL_H=150`, `FPS=60`, `MAX_DT=0.05`, `CAR_W=40`, `CAR_H=20`, `START_X=60`, `START_Y=60`, `MAX_SPEED=200.0`, `MAX_ACCEL=120.0`, `BRAKE_FORCE=2.5`, `FRICTION=60.0`, `MAX_STEER_RATE` (radians, `radians(120)`), `STEER_SMOOTHING=8.0`, `SENSOR_ANGLES` (radians of -60..60 step 30), `MAX_SENSOR_DISTANCE=200`, `SENSOR_STEP=4`, `SENSOR_HIT_DIST=2`, `STATE_DIM=6`, `ACTION_DIM=5`, `STEER_MAP`, `GAMMA`, `LR`, `BATCH_SIZE`, `MEMORY_SIZE`, `TARGET_UPDATE_STEPS`, `EPS_START=1.0`, `EPS_END=0.05`, `EPS_DECAY=0.9998`, `STALL_LIMIT=120`, `STALL_DISTANCE_EPS=0.05`, `MODEL_PATH="model.pt"`, plus reward weights `R_FORWARD`, `R_SLOW`, `R_FRONT_NEAR`, `R_FAST_NEAR`, `R_CRASH=-10.0`, `R_STALL=-5.0`, and colors.
- Produces: `tests/conftest.py` sets `os.environ["SDL_VIDEODRIVER"]="dummy"` before pygame imports.

- [ ] **Step 1: Write `tests/test_config.py::test_config_matches_spec`** asserting `len(SENSOR_ANGLES)==5`, `STATE_DIM==6`, `ACTION_DIM==len(STEER_MAP)==5`, `R_CRASH<0`, `R_STALL<0`, `EPS_END<EPS_START`.
- [ ] **Step 2: Run** `pytest tests/test_config.py -v`. Expected: FAIL (no module).
- [ ] **Step 3: Create** `requirements.txt` (`pygame`, `numpy`, `torch`, `pytest`), the package, `config.py` with the values above, and `conftest.py`.
- [ ] **Step 4: Run** `pip install -r requirements.txt` then `pytest tests/test_config.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: scaffold package and config`.

### Task 2: World (geometry, sensors, collisions)

**Files:**
- Create: `car_sim/world.py`, `tests/test_world.py`

**Interfaces:**
- Consumes: config constants; a `car` object exposing `x`, `y`, `angle` (Task 3 provides `Car`; tests use a tiny stand-in with those attributes).
- Produces:
  - `point_segment_distance(p: tuple[float,float], a: tuple[float,float], b: tuple[float,float]) -> float`
  - `class World` with `obstacles: list[tuple[tuple,tuple]]`; `add_segment(a, b) -> None`; `clear() -> None`; `cast_ray(x: float, y: float, angle: float) -> float` (distance in pixels, at most `MAX_SENSOR_DISTANCE`); `sensor_distances(car) -> list[float]` (one per `SENSOR_ANGLES`, relative to `car.angle`); `collided(car) -> bool`; `handle_mouse(pos, pressed: bool) -> None` (adds connected segments inside the game area, resets on release); `draw(surface) -> None`.

- [ ] **Step 1: Write tests**: `test_distance_projection_inside` (point (5,3), segment (0,0)-(10,0) -> 3.0); `test_distance_before_a` ((-3,4) -> 5.0); `test_distance_past_b` ((13,4) -> 5.0); `test_zero_length_segment` (segment (2,2)-(2,2), point (5,6) -> 5.0); `test_ray_hits_wall` (empty world, ray from (700,300) angle 0 returns less than `MAX_SENSOR_DISTANCE` and at most 100); `test_ray_hits_obstacle` (vertical segment at x=300, ray from (200,300) angle 0 returns about 100, within one `SENSOR_STEP`); `test_ray_clear_returns_max`; `test_collision_edge` (car at x=5); `test_collision_obstacle`; `test_no_collision_open_space`; `test_click_without_drag_is_safe` (press then release at the same pixel, then `cast_ray` and `collided` run without error); `test_clear_removes_obstacles`.
- [ ] **Step 2: Run** `pytest tests/test_world.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the signatures above. Ray marching: step `SENSOR_STEP` px from the origin, return the traveled distance at the first out-of-bounds point or point within `SENSOR_HIT_DIST` of any segment. `collided`: center closer than half the car width/height to an edge, or closer than `CAR_W / 2` to any segment. `draw` converts all coordinates to `int`, thickness 3.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: world geometry, sensors, collisions`.

### Task 3: Car physics

**Files:**
- Create: `car_sim/car.py`, `tests/test_car.py`

**Interfaces:**
- Produces: `class Car` with `x, y, angle, speed, ang_vel`; `reset() -> None` (spawn at `START_X, START_Y`, angle 0, speeds 0); `step(dt: float, throttle: float, brake: float, steer: float) -> float` (returns distance moved; clamps `dt` to `MAX_DT`); `draw(surface, color) -> None` (alpha surface, rotated by `-degrees(angle)`).

- [ ] **Step 1: Write tests**: `test_accelerates` (throttle 1 for 1 s: speed > 0 and <= `MAX_SPEED`); `test_speed_clamped` (long throttle never exceeds `MAX_SPEED`); `test_friction_slows` (no throttle: speed decreases, never below 0); `test_brake_stronger_than_friction` (braking stops faster than coasting from the same speed); `test_steering_is_smooth` (first frame of full steer: `ang_vel` less than `MAX_STEER_RATE`, converging toward it); `test_returns_distance_moved`; `test_dt_clamped` (`step(10.0, 1, 0, 0)` moves no more than one `MAX_DT` step would); `test_reset`.
- [ ] **Step 2: Run** `pytest tests/test_car.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the spec's Car physics (acceleration minus `BRAKE_FORCE`-scaled braking, friction only when throttle is 0, steering smoothing toward the target angular velocity, position along `cos`/`sin` of angle).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: car physics`.

### Task 4: UI and human mode

**Files:**
- Create: `car_sim/ui.py`, `car_hum.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes: `Car`, `World`, config.
- Produces: `draw_panel(surface, font, lines: list[str]) -> None`; `draw_steering_wheel(surface, steer: float) -> None`; `draw_pedal(surface, rect: tuple, value: float, color, label: str, font) -> None`; `car_hum.main() -> None`; `car_hum.read_keys(keys) -> tuple[float, float, float]` returning `(throttle, brake, steer)` (Up/W, Down/S, Left/A = -1, Right/D = +1).

- [ ] **Step 1: Write `tests/test_ui.py`**: `test_ui_draws_without_error` (headless surface, call all three draw functions with values 0, 0.5, 1, -1); `test_read_keys` using a dict-like stand-in for `pygame.key.get_pressed()` (Up -> throttle 1; Left -> steer -1; Left and Right together -> steer 0).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `ui.py` and `car_hum.py`: game loop at `FPS`, mouse drawing through `world.handle_mouse`, `V` toggles sensor rays (line plus end circle), `C` clears, `R` resets after a crash (also resetting score), crash sets game over and freezes updates, score rises only when the distance moved in a frame exceeds 1 px, panel shows score, max score, controls, and a red game-over message. Guard with `if __name__ == "__main__": main()`.
- [ ] **Step 4: Run** the tests. Then a headless smoke run: `SDL_VIDEODRIVER=dummy timeout 3 python car_hum.py`. Expected: tests PASS; the script runs until the timeout with no traceback.
- [ ] **Step 5: Commit** `feat: UI panel and human-driven simulator`.

### Task 5: Driving environment (state, controller, reward)

**Files:**
- Create: `car_sim/driving_env.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: `Car`, `World`, config.
- Produces:
  - `compute_reward(collided: bool, speed_norm: float, front_norm: float, brake: float, moved: float) -> float` (pure function; weights from config).
  - `class DrivingEnv` with `__init__(world: World | None = None)`, attributes `car`, `world`; `reset() -> numpy.ndarray` (float32, shape (6,)); `get_state() -> numpy.ndarray`; `step(action: int, dt: float = 1/60) -> tuple[numpy.ndarray, float, bool]`; `last_controls: tuple[float, float, float]` (throttle, brake, steer) for the UI.

- [ ] **Step 1: Write tests**: `test_state_shape_and_range` (shape (6,), dtype float32, all values in [0, 1]); `test_reward_crash_is_negative`; `test_reward_forward_positive` (no crash, moving, front clear, no braking > 0); `test_reward_near_obstacle_lower_than_clear`; `test_step_returns_triplet`; `test_stall_ends_episode` (action 2 with zero throttle forced by an obstacle directly in front: after `STALL_LIMIT` steps of near-zero movement `done` is True and the final reward is `R_STALL` or lower); `test_crash_ends_episode_negative` (car driven into an obstacle: `done` True, reward < 0); `test_reset_with_obstacle_on_spawn` (obstacle covering the spawn: `reset()` returns a state, and the first `step` returns `done=True`); `test_steer_map_used` (action 0 turns the car left, action 4 turns it right).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** the controller exactly as in the spec (steering scaled down with speed; brake from the front sensor clipped to [0, 1]; throttle cut above a small brake threshold; throttle reduced during sharp steering) and the reward terms and episode-end rules from the spec. Stall counter resets whenever `moved >= STALL_DISTANCE_EPS`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: driving environment with rewards and stall detection`.

### Task 6: Double DQN (PyTorch)

**Files:**
- Create: `car_sim/dqn.py`, `tests/test_dqn.py`

**Interfaces:**
- Consumes: config.
- Produces:
  - `class QNetwork(torch.nn.Module)`: `forward(x: Tensor[B,6]) -> Tensor[B,5]`.
  - `class ReplayBuffer`: `__init__(capacity: int)`, `push(state, action: int, reward: float, next_state, done: bool) -> None`, `sample(n: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]` (states float32 [n,6], actions int64 [n], rewards float32 [n], next_states float32 [n,6], dones float32 [n]), `__len__`.
  - `class DoubleDQNAgent`: `__init__(device: str = "cpu")`, `act(state, epsilon: float) -> int`, `remember(...)` (forwards to the buffer), `train_step() -> float | None` (None until the buffer holds `BATCH_SIZE`), `update_target() -> None`, `save(path: str, meta: dict) -> None`, `load(path: str) -> dict` (returns the meta dict; raises `ValueError` with a readable message when the file is missing keys or unreadable).

- [ ] **Step 1: Write tests**: `test_output_shape`; `test_buffer_respects_capacity`; `test_sample_shapes_and_dtypes`; `test_train_step_none_when_small`; `test_train_step_changes_online_weights`; `test_target_unchanged_until_update` then `test_update_target_copies`; `test_double_dqn_target` (hand-built: with a stub online net choosing action 1 and a stub target net valuing action 1 at 2.0, the target for reward 1.0, `GAMMA`, not done is `1.0 + GAMMA*2.0`; for done it is `1.0`; test the target-building helper `compute_targets(...)` directly); `test_act_greedy_when_epsilon_zero` and `test_act_random_when_epsilon_one` (many calls cover more than one action); `test_save_load_roundtrip` (same outputs after load; meta returned); `test_load_corrupt_file_raises_valueerror` (write garbage bytes to a temp path).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** with Adam, MSE loss, vectorized batch math: `next_actions = online(next).argmax(1)`, `next_q = target(next).gather(1, next_actions)`, `targets = r + GAMMA * next_q * (1 - done)` under `torch.no_grad()`. Expose that as `compute_targets(online, target, rewards, next_states, dones) -> Tensor`. Save uses `torch.save` of online and target state dicts, optimizer state, and meta.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: Double DQN agent in PyTorch`.

### Task 7: AI-driven simulator and end-to-end check

**Files:**
- Create: `self_driving_car_ai.py`, `tests/test_ai_loop.py`

**Interfaces:**
- Consumes: `DrivingEnv`, `DoubleDQNAgent`, `World`, UI helpers, config.
- Produces: `train(headless: bool = False, max_steps: int | None = None, model_path: str = MODEL_PATH) -> dict` (returns `{"episodes": int, "steps": int, "max_score": float}`); `main() -> None` calling `train()`.

- [ ] **Step 1: Write tests**: `test_short_headless_training_runs` (`train(headless=True, max_steps=2000, model_path=tmp)` completes, steps equal 2000, and a model file exists afterward); `test_resume_from_saved_model` (a second short run with the same path loads without error and continues the episode count); `test_corrupt_model_starts_fresh` (garbage file at `model_path`: `train` prints one message containing "starting fresh" via `capsys` and completes); `test_window_close_saves_model` (post a `pygame.QUIT` event in a short non-headless dummy-driver run: the model file exists afterward).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** the loop from the spec: aqua-blue car; epsilon-greedy action; `env.step`; `remember`; `train_step`; epsilon *= `EPS_DECAY` down to `EPS_END`; `update_target()` every `TARGET_UPDATE_STEPS`; on episode end record the score, update max and the 20-episode average, print `episode, score, max, avg`, reset. Panel shows episode, score, max, 20-episode average, epsilon. Mouse drawing and `C` / `V` keys work as in human mode. Save every 5 episodes and on exit (including `QUIT`). Headless mode skips all drawing and the frame limiter.
- [ ] **Step 4: Run** `pytest tests/test_ai_loop.py -v` and then the whole suite `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: self-driving simulator with live Double DQN training`.

### Task 8: README and final verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** covering what the project is, setup (`python -m venv .venv`, activate, `pip install -r requirements.txt`), the two run commands (`python car_hum.py`, `python self_driving_car_ai.py`), controls, where `config.py` values live, that the model saves to `model.pt` and resumes automatically (delete the file to start over), and that early training looks random while epsilon is high.
- [ ] **Step 2: Verify** by running `pytest -v` and both scripts headless with `SDL_VIDEODRIVER=dummy timeout 5 python <script>`. Expected: all tests pass; both scripts run until the timeout without a traceback.
- [ ] **Step 3: Package** the project (excluding `.git`, `__pycache__`, `.pytest_cache`, `model.pt`) into `self-driving-car.zip` and send it to the user.
- [ ] **Step 4: Commit** `docs: add README`.
