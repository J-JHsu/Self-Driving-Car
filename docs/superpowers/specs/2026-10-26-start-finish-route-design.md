# Start and Finish Points with a Route Compass: Design

Date: 2026-10-26
Status: awaiting user review
Builds on: `2026-10-25-car-simulator-design.md` (the existing simulator and Double DQN agent)

## Goal

Let the user place a **start** and a **finish** on a bigger map. In human mode the user drives from start to finish. In AI mode the agent learns to drive from start to finish around drawn walls. Dead-end pockets and mazes must not trap the guidance.

## Decisions (agreed with user)

- **Approach A+ ("route compass"):** the program computes the shortest route around the walls to the finish (flood fill on a coarse grid). The network is told which way the route points and how far is left. The network still learns the driving itself (steering, wall avoidance, momentum).
- **Map size:** 1400x800 game area plus the 150 px panel (window 1400x950), set in `config.py` so it can be changed.
- **Placing points:** press `1`, then left-click to place the start; press `2`, then left-click to place the finish. Works in both programs. The start and finish always exist (defaults below); there is no survive-only mode any more.
- The old `model.pt` will not load (state size changes 6 -> 8). `train()` already prints "starting fresh" and continues; the next save overwrites the old file.

## Changes to `car_sim/config.py`

- `GAME_W = 1400`, `GAME_H = 800`, `PANEL_H = 150`.
- `START_X, START_Y = 100, 400`; `GOAL_X, GOAL_Y = 1300, 400` (defaults).
- `GRID_CELL = 10`, `WALL_CLEARANCE = 20` (walls are thickened by this many px on the route grid, matching the car's collision radius), `GOAL_RADIUS = 30`.
- `STATE_DIM = 8`, `ROUTE_DIST_SCALE = 1600`.
- `R_PROGRESS = 0.1` (per pixel of route progress), `R_GOAL = 20.0`, `R_TIMEOUT = -5.0`, `MAX_EPISODE_STEPS = 3000`. `R_FORWARD` is removed (replaced by progress). `R_CRASH = -10`, `R_STALL = -5` and the proximity weights are unchanged.
- `START_HEADING_JITTER = radians(30)`. `RANDOM_SPAWN` stays but defaults to **False** (start point is used); when True, episodes start at a random free spot that has a route.
- Add `START_MARKER_COLOR`, `GOAL_MARKER_COLOR`, `ROUTE_COLOR`.

## Units

### `car_sim/world.py` (changed)
- New attributes `start: tuple[float, float]`, `goal: tuple[float, float]` (defaults from config) and `version: int`.
- `set_start(pos) -> None`, `set_goal(pos) -> None`: ignore positions outside the game area; bump `version`.
- `add_segment` and `clear` also bump `version`. `clear()` removes walls only; start and goal stay.
- Everything else (rays, collisions, mouse drawing, dot-on-click) is unchanged.

### `car_sim/route.py` (new)
- `class RouteField`:
  - `ensure(world) -> bool`: rebuild only if `world.version` changed since the last build; returns whether the start has a route.
  - `distance(x, y) -> float`: route length in pixels to the finish; `math.inf` when unreachable.
  - `direction(x, y) -> float | None`: absolute angle (radians) of the steepest-descent step toward the finish, or `None` if no route.
  - `reachable(x, y) -> bool`.
  - `path(x, y, max_points=400) -> list[tuple[float, float]]`: the route as a polyline, for the overlay.
- Algorithm: grid of `GAME_H/GRID_CELL` x `GAME_W/GRID_CELL` cells. Mark cells within `WALL_CLEARANCE` of any wall segment or of the map border as blocked; the finish cell is always free. Dijkstra from the finish over 8 neighbors (diagonal cost sqrt(2)), no diagonal moves between two blocked orthogonal neighbors. If the query point's cell is blocked, use the nearest free cell within 3 cells; if none, treat as unreachable.
- Cost: about 11,200 cells; rebuilds only when walls or points change. While the mouse is drawing, this means a rebuild per frame, which can feel slightly laggy on a big map; acceptable.

### `car_sim/car.py` (changed)
- `reset(x=None, y=None, angle=0.0)`: defaults to the config start. Physics unchanged.

### `car_sim/driving_env.py` (changed)
- `DrivingEnv(world=None, random_spawn=None)` owns a `RouteField`.
- State (float32, 8): 5 sensors / `MAX_SENSOR_DISTANCE`, speed / `MAX_SPEED`, route direction relative to the car's heading divided by pi (range -1..1; 0 = straight ahead), `min(1, route_distance / ROUTE_DIST_SCALE)`. With no route the last two are 0 and 1.
- `reset()`: car at `world.start`, heading = route direction + uniform jitter within `START_HEADING_JITTER` (heading 0 if there is no route); with `random_spawn`, a random free spot that has a route (falls back to the start after 50 tries). Clears `steps` and `outcome`.
- `step(action)` returns `(state, reward, done)` and sets `env.outcome` to one of `""`, `"goal"`, `"crash"`, `"stall"`, `"timeout"`.
  - `progress = previous_route_distance - route_distance` (pixels, can be negative); reward adds `R_PROGRESS * progress`.
  - Reaching within `GOAL_RADIUS` of the finish: `R_GOAL`, done.
  - Crash and stall as before; `MAX_EPISODE_STEPS` reached: `R_TIMEOUT`, done.
  - Controller rules (steering, braking, throttle) and the proximity penalties are unchanged.
- `route_ok: bool` property: whether the start currently has a route.
- `compute_reward(collided, speed_norm, proximity_norm, brake, progress, reached) -> float`.

### `car_sim/ui.py` (changed)
- `draw_markers(surface, world, placing)`: start (green circle labeled S) and finish (checkered/red circle labeled G); a hint in the panel while placing.
- `draw_route(surface, world_route_points)`: the route polyline, toggled with `P`.
- `handle_placement_event(event, world, mode) -> mode`: keys `1`/`2` choose the mode, the next left-click sets the point and ends the mode; `Esc` cancels. While a mode is active the click is not used for drawing.

### `car_hum.py` (changed)
- Same controls plus `1`/`2` placement and `P`. Car spawns at the start. Reaching the finish ends the run with a "YOU WIN" message and the elapsed time; `R` restarts. Crash behaves as before.

### `self_driving_car_ai.py` (changed)
- Same controls plus `1`/`2` placement and `P`. Panel adds goals reached out of the last 20 episodes. The per-episode console line includes the outcome.
- If there is no route, training pauses (the window keeps running) and the panel says "No route from start to finish: move a point or open a gap". It resumes automatically when a route exists.
- Placing a point or changing walls mid-episode ends nothing by itself; the next `reset()` uses the new start.
- `train()` still returns episodes, steps and max score, plus `goals` (number of episodes that ended at the finish).

## Error handling

- No route: described above (message, no crash, training pauses).
- Start or finish placed inside a wall or too close to one: allowed; the route check reports no route if the finish cell cannot be reached; a start inside a wall crashes immediately and the panel explains if there is no route.
- Corrupt or old-shape `model.pt`: already handled (message, start fresh).

## Testing (headless pytest)

- **RouteField:**
  - open map: direction points at the finish and distance is about the straight-line distance;
  - U-shaped pocket between a car in the pocket and the finish: the direction leaves the pocket (not straight through the wall) and the distance is far larger than the straight line;
  - fully walled-off finish: `distance` is inf, `reachable` is False, `direction` is None;
  - a gap narrower than the car is blocked, a wide gap is open;
  - `ensure` rebuilds only when `world.version` changes (counted);
  - a query point inside a blocked cell uses the nearest free cell.
- **World:** `set_start/set_goal` change positions, bump `version`, ignore out-of-bounds; `clear()` keeps the points.
- **Env:** state has shape 8 and sensible ranges; moving along the route gives a positive reward and moving away a negative one; reaching the finish ends the episode with `R_GOAL` and `outcome == "goal"`; the time limit ends with `R_TIMEOUT`; start heading is within the jitter of the route direction; `route_ok` is False with a blocked finish; random spawn stays free and routable.
- **UI and placement:** `1`/`2` plus click set the points and do not add walls; `Esc` cancels; `draw_markers` and `draw_route` run headless.
- **Programs:** `car_hum` win condition via `HumanGame`; `train()` short headless run, resume, corrupt/old model, window-close save (existing tests updated for 8 inputs); both scripts run headless 5 s with no traceback.
- **Learning check (reported, not a pass/fail gate):** headless training on a map with a U-shaped pocket between start and finish; report the fraction of the last 50 episodes that reach the finish, the average episode length, and what fraction of greedy rollouts reach the finish. These numbers go in the README and the final message as measured, whether good or bad.

## Out of scope

Multiple finishes or waypoints, moving walls, undoing individual walls, saving and loading maps, laps, and a "learn from the real map without the route" mode.

## Known limits

- The car has no memory; the route compass supplies the plan. If walls fully block the finish there is no route.
- Gaps narrower than about twice `WALL_CLEARANCE` (40 px) are treated as walls for routing.
- The route is on a 10 px grid, so it is approximate; the car can still crash on tight turns.
- A 1400x950 window may not fit a screen with high display scaling; change `GAME_W`, `GAME_H`, `PANEL_H` in `config.py`.
- Learning quality depends on training time and map; results are measured, not guaranteed.
