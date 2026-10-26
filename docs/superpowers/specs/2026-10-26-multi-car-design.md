# Multi-car training (vectorized environments) - design

Status: implemented 2026-10-26 with the assumptions below (per-transition counting, `NUM_CARS = 8`). The "Measurements to decide the defaults" have not been run.

## Goal

Train the same Double DQN faster in wall-clock time by driving several cars on the same map at once, with every car visible in the window. Single-car behavior stays available (`--cars 1`).

## Decisions already made with the user

- Approach: many `DrivingEnv` copies stepped in one loop, one shared network, one shared replay buffer (the standard "vectorized environment" pattern, like `DummyVecEnv`). Not separate processes, not a GPU.
- All cars visible on screen. Fast mode still skips most drawing.
- Cars are ghosts: they do not collide with each other.
- Number of cars is a setting in `config.py` and a command-line flag.
- Saved models stay compatible (state size 8 and the network do not change).
- Human mode (`car_hum.py`) is not touched.

## Assumptions (not yet confirmed)

- Default `NUM_CARS = 8`.
- Epsilon, the target-network update and the step counters count transitions (one per car per loop), so the exploration schedule means the same thing as before. This is a choice, see "Open questions".

## Design

### 1. Shared route (`car_sim/driving_env.py`)
`DrivingEnv.__init__` gets an optional `route=None` argument. If given, the env uses that `RouteField`/`StraightField` instead of building its own. Today each env owns one (line ~57), so N cars would otherwise rebuild the Dijkstra field N times after every wall edit. `ensure(world)` is already cached on `world.version`, so a shared one rebuilds once.

### 2. `VecDrivingEnv` (new file `car_sim/vec_env.py`)
- `__init__(world, num_cars, use_route)`: one world, one route object, `num_cars` `DrivingEnv`s.
- `reset()`: resets all, returns a list of states.
- `step(actions)`: steps each env with its action. If an env finishes (goal, crash, stall, timeout) it records `{outcome, score}` in a `finished` list and resets itself immediately, so the other cars keep driving. Returns `states_next, rewards, dones` plus the finished list. The transition stored for a finished car uses the terminal next-state, not the reset state.
- `check_route()`: calls the shared route once; `route_ok` is shared.
- `cars`: the list of `Car` objects for drawing.

### 3. Batched action choice (`car_sim/dqn.py`)
`DoubleDQNAgent.act_batch(states, epsilon)`: one forward pass on a `(N, 8)` tensor, then each car independently takes a random action with probability epsilon. `act` stays for the single-car path and tests. Training (`train_step`) is unchanged.

### 4. Training loop (`self_driving_car_ai.py`)
Per loop:
1. `actions = agent.act_batch(states, epsilon)`
2. `vec.step(actions)`
3. `remember` every transition
4. run `UPDATES_PER_LOOP` calls to `train_step()`
5. decay epsilon once per transition, and call `update_target()` each time the transition count crosses a multiple of `TARGET_UPDATE_STEPS`
6. for each finished episode: update episode, score, max, avg, goal counters, print the usual line, save every `SAVE_EVERY_EPISODES` episodes

The pause-when-no-route logic keeps working on the shared `route_ok`. `train()` gets a `num_cars` argument and keeps returning the same summary dict.

### 5. Drawing and controls
- All cars drawn each frame; the first car is the "focus car" and is the only one that shows sensors. The rest are drawn semi-transparent.
- The panel shows the same stats plus `Cars: N`. Controls are unchanged (`F` fast, `S` sensors, `P` route, `Space` pause, placement and erase).
- Fast mode: draws every Nth loop as now.

### 6. Config and command line
New values in `config.py`: `NUM_CARS = 8`, `UPDATES_PER_LOOP = 2`. `python self_driving_car_ai.py --cars 4` overrides; `--cars 1` reproduces today's behavior. `--no-route` combines with it.

## What does not change
State layout, rewards, network, saved-model format, route algorithm, human mode.

## Testing (TDD, red then green)
- `VecDrivingEnv` with N=1 produces the same transitions as a bare `DrivingEnv` given the same actions and seeds.
- N cars share one route object and the route is built once after N resets (`_route_builds`).
- A crashing car resets by itself and the others are unaffected; the finished list reports outcome and score.
- A terminal transition stores the terminal next-state.
- `act_batch` returns N valid actions; with epsilon 0 it matches `act` per state; with epsilon 1 it is random.
- Epsilon decay and target updates count transitions: after K loops with N cars, epsilon equals `EPS_START * EPS_DECAY**(K*N)`.
- A headless `train(num_cars=3, max_steps=...)` run completes and returns the summary dict; a changed wall triggers one route rebuild shared by all cars.
- UI test: drawing N cars does not raise and the panel text fits.
- Existing 130 tests keep passing.

## Measurements to decide the defaults
Not assumptions, to be run before settling `NUM_CARS` and `UPDATES_PER_LOOP`:
- Wall-clock time to reach a fixed goal rate on the default map for `--cars` 1, 4, 8 at `UPDATES_PER_LOOP` 1, 2, 4, 8.
- Use the greedy 100-run check from the earlier learning test (`learn_check.py`-style), more than one run per setting because earlier results varied between checkpoints.
- Report goals per minute and greedy success. If 8 cars is not clearly faster for the same quality, say so and keep the default at 1.

## Risks
- Fewer updates per experience can slow learning per step; that is what `UPDATES_PER_LOOP` and the measurements are for.
- The 50k replay buffer fills N times faster, so old experience is dropped sooner. May need a larger `MEMORY_SIZE`.
- Cars started together and driving similar paths give correlated data; the start heading jitter and epsilon help, but this is not measured.
- Sensor and car updates cost scales with N; the speed-up is below N.

## Out of scope
Car-to-car collision, multiple processes, GPU use, changing the state or rewards, human-mode changes.

## Open questions for the user
1. Count epsilon and target updates per transition (assumed) or per loop?
2. Default `NUM_CARS` of 8, or smaller?
