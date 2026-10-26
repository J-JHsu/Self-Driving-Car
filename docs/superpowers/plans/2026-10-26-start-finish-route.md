# Start/Finish + Route Compass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a placeable start and finish on a 1400x800 map, with a route compass (shortest route around walls) feeding the DQN, in both programs.

**Architecture:** `World` holds start, finish and a change counter. A new `RouteField` (grid flood fill from the finish) answers distance and direction queries and rebuilds only when the world changes. `DrivingEnv` gets an 8-number state and progress/goal rewards. Both entry scripts add placement keys, markers and a route overlay.

**Tech Stack:** Python 3.10+, pygame (pygame-ce works), numpy, torch, pytest. Existing code in `car_sim/`.

**Spec:** `docs/superpowers/specs/2026-10-26-start-finish-route-design.md` (builds on `2026-10-25-car-simulator-design.md`)

## Global Constraints

- Map 1400x800 + panel 150 (`GAME_W`, `GAME_H`, `PANEL_H` in config). Defaults: start (100, 400), finish (1300, 400).
- Grid cell 10 px; walls thickened by `WALL_CLEARANCE = 20` px; `GOAL_RADIUS = 30`.
- State is 8 float32 values: 5 sensors / `MAX_SENSOR_DISTANCE`, speed / `MAX_SPEED`, route direction relative to heading / pi (range -1..1), `min(1, route_distance / ROUTE_DIST_SCALE)` with `ROUTE_DIST_SCALE = 1600`. No route: last two are 0 and 1.
- Rewards: `R_PROGRESS = 0.1` per pixel of route progress (negative when moving away), `R_GOAL = 20.0`, `R_TIMEOUT = -5.0`, `MAX_EPISODE_STEPS = 3000`; `R_CRASH = -10`, `R_STALL = -5` unchanged; `R_FORWARD` removed.
- Placement: key `1` then left-click = start; key `2` then left-click = finish; `Esc` cancels; `P` toggles the route overlay. Start and finish always exist.
- `START_HEADING_JITTER = radians(30)`; `RANDOM_SPAWN` defaults to False.
- Old 6-input `model.pt` must not crash: one "starting fresh" message.
- All tests headless (`SDL_VIDEODRIVER=dummy`); run the whole suite with `python3 -m pytest -q` after every task.

## Review Focus

- Start equal to (or within 30 px of) the finish: the first step ends the episode as a goal; no loop.
- A wall drawn mid-training that fully blocks the finish: windowed training pauses with a message and resumes when a route exists; headless returns instead of hanging.
- A route rebuild must stay fast on the full 1400x800 grid (under 1 s with 200 segments) because it reruns while the mouse is drawing.
- Start or finish placed inside or touching a wall: no exception anywhere; the program reports no route or crashes the episode normally.
- A saved model with the old input size: one "starting fresh" message, training continues.

---

### Task 1: Config, window size and Car.reset parameters

**Files:**
- Modify: `car_sim/config.py`, `car_sim/car.py`, `tests/test_world.py`, `tests/test_env.py`, `tests/test_ui.py`, `tests/test_car.py`, `tests/test_config.py`

**Interfaces:**
- Produces config constants: `GAME_W=1400`, `GAME_H=800`, `START_X=100`, `START_Y=400`, `GOAL_X=1300`, `GOAL_Y=400`, `GRID_CELL=10`, `WALL_CLEARANCE=20`, `GOAL_RADIUS=30`, `ROUTE_DIST_SCALE=1600`, `R_PROGRESS=0.1`, `R_GOAL=20.0`, `R_TIMEOUT=-5.0`, `MAX_EPISODE_STEPS=3000`, `START_HEADING_JITTER` (radians(30)), `RANDOM_SPAWN=False`, `START_MARKER_COLOR`, `GOAL_MARKER_COLOR`, `ROUTE_COLOR`. Remove `R_FORWARD` later in Task 4 (kept until the reward function changes). Do **not** change `STATE_DIM` here.
- Produces: `Car.reset(x: float | None = None, y: float | None = None, angle: float = 0.0) -> None` (None means config start).

- [ ] **Step 1: Write failing tests.** In `tests/test_config.py::test_map_and_defaults`: `GAME_W==1400`, `GAME_H==800`, `(START_X,START_Y)==(100,400)`, `(GOAL_X,GOAL_Y)==(1300,400)`, `GOAL_RADIUS==30`, `RANDOM_SPAWN is False`. In `tests/test_car.py::test_reset_to_given_position`: `Car().reset(500, 300, 1.0)` sets `x==500`, `y==300`, `angle==1.0`, speed and `ang_vel` 0.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_config.py tests/test_car.py -q`. Expected: the two new tests FAIL.
- [ ] **Step 3: Implement** the config values and `Car.reset(x, y, angle)`.
- [ ] **Step 4: Fix existing tests that hard-code the old map:** `test_world.py::test_ray_hits_wall` uses origin x=1300 (distance to wall <= 100); `test_env.py` and `test_ui.py` walls that were `(x, 0)-(x, 200/300/600)` become `(x, 0)-(x, config.GAME_H)`; `test_env.py` crash/stall/spawn walls are relative to `config.START_X` (keep) but span the full height.
- [ ] **Step 5: Run** `python3 -m pytest -q`. Expected: all PASS.
- [ ] **Step 6: Commit** `feat: bigger map, new defaults, Car.reset position`.

### Task 2: World start, finish and version

**Files:**
- Modify: `car_sim/world.py`, `tests/test_world.py`

**Interfaces:**
- Produces on `World`: attributes `start: tuple[float, float]`, `goal: tuple[float, float]`, `version: int`; `set_start(pos) -> None`; `set_goal(pos) -> None` (out-of-bounds positions are ignored); `add_segment` and `clear` bump `version`; `clear()` keeps start and goal.

- [ ] **Step 1: Write failing tests:** `test_defaults_are_config_values`; `test_set_start_and_goal_bump_version` (positions change and `version` increases each time); `test_out_of_bounds_points_ignored` (`(1500, 10)` and `(-5, 20)` change nothing and do not bump); `test_add_segment_and_clear_bump_version`; `test_clear_keeps_start_and_goal`.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_world.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per the interface; version starts at 0.
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: world start/finish points and change counter`.

### Task 3: RouteField

**Files:**
- Create: `car_sim/route.py`, `tests/test_route.py`

**Interfaces:**
- Consumes: `World` (`obstacles`, `goal`, `version`), config grid constants.
- Produces `class RouteField`: attribute `builds: int` (count of rebuilds); `ensure(world) -> bool` (rebuilds only if `world.version` or the goal changed since the last build; returns whether `world.start` has a route); `distance(x: float, y: float) -> float` (`math.inf` if unreachable); `direction(x, y) -> float | None` (absolute angle of the steepest-descent step, `None` if unreachable); `reachable(x, y) -> bool`; `path(x, y, max_points: int = 400) -> list[tuple[float, float]]`.

- [ ] **Step 1: Write failing tests** (empty `World` has goal (1300, 400)): `test_open_map_direction_and_distance` (from (100,400): distance within 5% of 1200, `abs(direction) < 0.2`); `test_u_pocket_routes_out_the_open_side` (walls `(600,300)-(600,500)`, `(400,300)-(600,300)`, `(400,500)-(600,500)`, pocket open to the left; from (500,400): `abs(direction) > pi/2` and `distance > 1000`); `test_walled_off_goal_is_unreachable` (wall `(1200,0)-(1200,800)`: `distance` is inf, `reachable` False, `direction` None, `ensure` returns False); `test_narrow_gap_blocked_wide_gap_open` (wall x=700 with a 30 px gap at y 385-415: start side unreachable; with an 80 px gap at 360-440: reachable); `test_ensure_rebuilds_only_on_change` (`builds` 1 after two calls; 2 after `add_segment`; 3 after `set_goal`); `test_point_in_blocked_cell_uses_nearest_free` (wall `(200,400)-(800,400)`; `distance(500, 405)` is finite); `test_path_follows_route_to_goal` (open map: first point near the query, last point within `GOAL_RADIUS` of the goal, length <= `max_points`); `test_build_is_fast` (200 random segments: one `ensure` call under 1.0 s).
- [ ] **Step 2: Run** `python3 -m pytest tests/test_route.py -q`. Expected: FAIL (no module).
- [ ] **Step 3: Implement** `RouteField`. Algorithm (spec is binding): numpy grid `GAME_H/GRID_CELL` x `GAME_W/GRID_CELL`; mark blocked every cell whose center is closer than `WALL_CLEARANCE` to any segment (use `point_segment_distance`-style math vectorized, or dilate a rasterized segment mask) and every cell within `WALL_CLEARANCE` of the border; the goal cell stays free; Dijkstra with `heapq` over 8 neighbors (diagonal cost sqrt(2) * `GRID_CELL`), skipping diagonal moves when both orthogonal neighbors are blocked. A query point in a blocked cell uses the nearest free cell within 3 cells, else unreachable. `direction` points from the query's free cell to its lowest-distance neighbor; at the goal cell it points at the goal.
- [ ] **Step 4: Run** `python3 -m pytest tests/test_route.py -q` then the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: RouteField (shortest route around walls)`.

### Task 4: DrivingEnv with route state and goal rewards

**Files:**
- Modify: `car_sim/driving_env.py`, `car_sim/config.py` (`STATE_DIM = 8`, remove `R_FORWARD`), `tests/test_env.py`, `tests/test_dqn.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `World.start/goal/version`, `RouteField`, `Car.reset(x, y, angle)`.
- Produces: `compute_reward(collided: bool, speed_norm: float, proximity_norm: float, brake: float, progress: float, reached: bool) -> float`; `DrivingEnv.route_ok: bool` (property); `DrivingEnv.outcome: str` (`""`, `"goal"`, `"crash"`, `"stall"`, `"timeout"`); `DrivingEnv.steps: int`; state of shape (8,) as in Global Constraints; `reset()` places the car at `world.start`, heading = route direction + uniform jitter within `START_HEADING_JITTER` (heading 0 if no route); with `random_spawn` a random free spot that has a route (fallback to the start after 50 tries).

- [ ] **Step 1: Update and add failing tests.** Replace hard-coded 6 with `config.STATE_DIM` in `test_env.py` and `test_dqn.py`; set `STATE_DIM == 8` in `test_config.py`. New in `test_env.py`: `test_state_has_route_values` (`state[6]` in [-1, 1], `state[7]` in [0, 1]; heading straight at the goal on an open map gives `abs(state[6]) < 0.1`); `test_progress_sign` (two fresh envs, one with `car.angle = 0`, one with `car.angle = pi`: after one step the first reward is higher); `test_reaching_goal_ends_episode` (`world.set_goal((START_X + 25, START_Y))`: first `step` returns `done` True, `outcome == "goal"`, reward >= `R_GOAL`); `test_start_equals_goal_ends_immediately` (same, goal == start); `test_timeout_ends_episode` (monkeypatch `config.MAX_EPISODE_STEPS = 5`, open map: not done for 4 steps, 5th returns `done` True, `outcome == "timeout"`, reward == `R_TIMEOUT`); `test_reset_heading_within_jitter` (50 resets: `abs(angle - route_direction)` <= `START_HEADING_JITTER`); `test_route_ok_false_when_finish_blocked_and_step_still_works` (wall `(1200,0)-(1200,800)`: `route_ok` False, `step` returns a triplet, `state[7] == 1.0`); `test_random_spawn_is_free_and_routable`; `test_start_inside_wall_does_not_raise`. Update `test_crash_ends_episode_negative` to also assert `outcome == "crash"` and `test_stall_ends_episode` to assert `outcome == "stall"`.
- [ ] **Step 2: Run** `python3 -m pytest tests/test_env.py tests/test_dqn.py tests/test_config.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the env per the spec. The sensor-cast-once-per-step behavior from the existing env stays (reuse the cached state; refresh the route with `ensure` once per step). Progress is previous route distance minus new route distance, 0 when either is infinite. Keep the controller rules and `proximity()` as they are. Order of end conditions: goal, crash, stall, timeout.
- [ ] **Step 4: Run** the full suite. Expected: PASS (AI-loop tests may need `STATE_DIM`-independent edits only; fix if they fail for that reason alone).
- [ ] **Step 5: Commit** `feat: env with route compass state and goal rewards`.

### Task 5: UI markers, route overlay, placement, human mode

**Files:**
- Modify: `car_sim/ui.py`, `car_hum.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes: `World`, `RouteField`.
- Produces: `draw_markers(surface: Surface, world: World, placing: str | None) -> None`; `draw_route(surface: Surface, points: list[tuple[float, float]]) -> None`; `handle_placement_event(event, world: World, mode: str | None) -> str | None` (key `1` -> `"start"`, key `2` -> `"goal"`, `Esc` -> `None`; a left `MOUSEBUTTONDOWN` inside the game area while a mode is active sets that point and returns `None`; clicks outside the game area are ignored and the mode stays); `HumanGame` gains `won: bool` and `elapsed: float`, starts at `world.start`, and `reset()` returns the car to `world.start`.

- [ ] **Step 1: Write failing tests:** `test_markers_draw_at_points` (pixel at the start center equals `START_MARKER_COLOR`; at the goal center equals `GOAL_MARKER_COLOR`); `test_draw_route_runs`; `test_placement_keys_and_click` (K_1 then click (300,300) -> `world.start == (300,300)`, mode `None`, no obstacles added; K_2 then click -> goal); `test_placement_escape_cancels`; `test_placement_click_in_panel_ignored_mode_stays`; `test_human_wins_at_finish` (`world.set_goal((START_X + 25, START_Y))`: first `update` sets `won` and `game_over`); `test_human_win_freezes_and_reset_restarts_from_start` (after a win, `update` changes nothing; `reset` puts the car at `world.start`, `won` False); `test_human_spawns_at_world_start` (`set_start((300, 300))` then `reset` puts the car there).
- [ ] **Step 2: Run** `python3 -m pytest tests/test_ui.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the UI helpers and `HumanGame` changes. In `car_hum.main`: route events through `handle_placement_event`; while a placement mode is active or on the frame a placement click is consumed, pass `pressed=False` to `world.handle_mouse`; `P` toggles the overlay (path from the car); show the panel hint while placing; on win show a "YOU WIN" message with `elapsed` seconds; `R` restarts after a win or crash.
- [ ] **Step 4: Run** the full suite and `SDL_VIDEODRIVER=dummy timeout 5 python3 car_hum.py`. Expected: tests PASS; the script runs until the timeout with no traceback.
- [ ] **Step 5: Commit** `feat: start/finish placement, markers, route overlay, human win`.

### Task 6: AI program

**Files:**
- Modify: `self_driving_car_ai.py`, `tests/test_ai_loop.py`

**Interfaces:**
- Consumes: `DrivingEnv` (`route_ok`, `outcome`), UI helpers from Task 5.
- Produces: `train(headless: bool = False, max_steps: int | None = None, model_path: str = MODEL_PATH, world: World | None = None) -> dict` returning `episodes`, `steps`, `max_score`, `goals` (episodes that ended with outcome `"goal"`), and `no_route` (True if a headless run stopped because the start has no route). Windowed runs pause training while `route_ok` is False and show "No route from start to finish: move a point or open a gap".

- [ ] **Step 1: Update and add failing tests.** Existing four tests keep their assertions (they now run with the route state). Add `test_result_has_goals_count` (short headless run: `0 <= goals <= episodes`); `test_headless_no_route_returns_instead_of_hanging` (`world` with `(1200,0)-(1200,800)`: `train(headless=True, max_steps=500, world=world)` returns with `no_route` True and `steps == 0`); `test_old_input_size_model_starts_fresh` (monkeypatch `config.STATE_DIM = 6`, create a `DoubleDQNAgent`, save to `model.pt`, undo the patch; `train(headless=True, max_steps=200, ...)` prints "starting fresh" once and returns `steps == 200`); `test_goal_reached_counts` (tiny world where the finish is 25 px from the start: a 300-step run ends several episodes with `goals >= 1`).
- [ ] **Step 2: Run** `python3 -m pytest tests/test_ai_loop.py -q`. Expected: new tests FAIL.
- [ ] **Step 3: Implement** the loop changes: placement keys and `P` overlay as in Task 5; panel adds "Reached finish: n of last 20 episodes" and the pause message; console line per episode includes the outcome; headless with no route prints one message and returns.
- [ ] **Step 4: Run** the full suite and `SDL_VIDEODRIVER=dummy timeout 5 python3 self_driving_car_ai.py`. Expected: PASS and no traceback.
- [ ] **Step 5: Commit** `feat: AI mode with start/finish, route compass and pause on no route`.

### Task 7: Learning check, README, review, package

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-10-26-start-finish-route-design.md` (measured results only if the spec needs a revision note)

- [ ] **Step 1: Learning check** (scratch script outside the repo, background run). Map: start (100, 400), finish (1300, 400), a U-shaped pocket `(600,300)-(600,500)`, `(400,300)-(600,300)`, `(400,500)-(600,500)` open to the left, plus two more walls; train headless 150k steps; report fraction of the last 50 episodes with outcome `"goal"`, average episode length, and the fraction of 20 greedy rollouts that reach the finish. Report numbers as measured. If the agent does not learn, diagnose causes (reward scale, controller, spawn) with a quick experiment before changing defaults, and ledger any change.
- [ ] **Step 2: Update `README.md`:** new controls (`1`/`2`, `P`), the route compass explanation, the window size setting, the measured results from Step 1, the old-model note, and Python/pygame-ce install notes (`pip install pygame-ce` if `pygame` fails to build).
- [ ] **Step 3: Verify:** `python3 -m pytest -q`; both scripts headless 5 s with no traceback.
- [ ] **Step 4: Final whole-branch review** with a fresh reviewer (Opus), giving it the Review Focus section; fix Critical and Important findings test-first.
- [ ] **Step 5: Package** `self-driving-car.zip` (excluding `.git`, caches, `.superpowers`, `model.pt`) and send it; commit `docs: README and results for start/finish`.
