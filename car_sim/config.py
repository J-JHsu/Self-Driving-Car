"""Every tunable value for the simulator and the agent lives here."""
import math

# --- Window -----------------------------------------------------------------
GAME_W = 1400
GAME_H = 800
PANEL_H = 170
FPS = 60
MAX_DT = 0.05  # frame times above this are clamped so lag never teleports the car

# --- Car --------------------------------------------------------------------
CAR_W = 40
CAR_H = 20
START_X = 100
START_Y = 400
GOAL_X = 1300
GOAL_Y = 400
MAX_SPEED = 120.0
MAX_ACCEL = 120.0
BRAKE_FORCE = 2.5  # braking is this many times stronger than acceleration
FRICTION = 60.0
MAX_STEER_RATE = math.radians(120)
STEER_SMOOTHING = 8.0

# --- Sensors ----------------------------------------------------------------
SENSOR_ANGLES = [math.radians(a) for a in (-60, -30, 0, 30, 60)]
MAX_SENSOR_DISTANCE = 200
SENSOR_STEP = 4
SENSOR_HIT_DIST = 2

# --- Drawing ----------------------------------------------------------------
ERASE_RADIUS = 12  # right-drag deletes walls this close to the cursor
MOUSE_MIN_SEGMENT = 3  # pixels the mouse must move before another segment is added

# --- Agent ------------------------------------------------------------------
STATE_DIM = 8  # 5 sensor distances + speed + route direction + route distance
ACTION_DIM = 5
STEER_MAP = [-1.0, -0.5, 0.0, 0.5, 1.0]
HIDDEN_SIZE = 64

GAMMA = 0.99
LR = 3e-4  # 1e-3 made the greedy policy swing between checkpoints (see README)
BATCH_SIZE = 64
MEMORY_SIZE = 50_000
TARGET_UPDATE_STEPS = 500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9998  # multiplied in every step (one step = one car's transition)

# --- Multi-car training -----------------------------------------------------
NUM_CARS = 8          # cars driving at once in training (--cars overrides; 1 = the old single car)
UPDATES_PER_LOOP = 2  # gradient steps per loop over all cars, capped at the number of cars

# --- Controller rules around the chosen steering ----------------------------
BRAKE_TRIGGER_DIST = 0.5   # front sensor (0-1) below this starts braking
BRAKE_ACTIVE_THRESHOLD = 0.1
SHARP_STEER = 0.75         # |steer| above this eases the throttle
SHARP_STEER_THROTTLE = 0.6
STEER_SPEED_SCALE = 0.0    # 0 = full steering at any speed; 0.5 = half steering at top speed
PROXIMITY_SIDE_MARGIN = 0.25  # side sensors count as 'near' this much sooner than the front
RANDOM_SPAWN = False       # True: AI episodes start at a random free spot that has a route
SPAWN_MARGIN = 60          # keep random spawns this far from the walls
SPAWN_TRIES = 50

# --- Route / goal -----------------------------------------------------------
GRID_CELL = 10             # route grid resolution in pixels
WALL_CLEARANCE = 24        # route never runs closer than this to a wall (the car crashes at 20)
WALL_SOFT_ZONE = 56        # inside this distance the route pays extra, so it prefers the middle
WALL_SOFT_COST = 1.5       # extra cost per pixel at the clearance line; fades to 0 at the soft zone edge
GOAL_RADIUS = 30           # reaching within this many pixels of the finish wins
ROUTE_DIST_SCALE = 1600    # route distance (px) that maps to 1.0 in the state
START_HEADING_JITTER = math.radians(30)
R_PROGRESS = 0.1           # per pixel of route progress (negative when moving away)
R_GOAL = 20.0
R_TIMEOUT = -5.0
MAX_EPISODE_STEPS = 3000

# --- Episode end ------------------------------------------------------------
STALL_LIMIT = 120
STALL_DISTANCE_EPS = 0.05

# --- Rewards ----------------------------------------------------------------
R_SLOW = -0.05        # per step when almost stopped
R_FRONT_NEAR = 1.0    # scales the penalty as the front obstacle gets close
R_FAST_NEAR = 2.0     # extra penalty when fast and close
R_CRASH = -10.0
R_STALL = -5.0
SLOW_SPEED_NORM = 0.1
FRONT_NEAR_NORM = 0.5
FAST_SPEED_NORM = 0.5
FAST_NEAR_NORM = 0.25
BRAKE_HARD = 0.5

# --- Files ------------------------------------------------------------------
MODEL_PATH = "model.pt"
MODEL_PATH_NO_ROUTE = "model_no_route.pt"  # the --no-route mode learns a different task
SAVE_EVERY_EPISODES = 5

# --- Colors -----------------------------------------------------------------
BG_COLOR = (25, 25, 30)
PANEL_COLOR = (40, 40, 50)
HUMAN_CAR_COLOR = (60, 200, 90)
AI_CAR_COLOR = (0, 200, 220)
GHOST_CAR_ALPHA = 110  # opacity (0-255) of the AI cars other than the focus car
OBSTACLE_COLOR = (230, 230, 230)
SENSOR_COLOR = (255, 200, 0)
TEXT_COLOR = (235, 235, 235)
GAME_OVER_COLOR = (230, 60, 60)
THROTTLE_COLOR = (60, 200, 90)
BRAKE_COLOR = (230, 60, 60)
WHEEL_COLOR = (200, 200, 210)
START_MARKER_COLOR = (60, 200, 90)
GOAL_MARKER_COLOR = (230, 60, 60)
ROUTE_COLOR = (90, 160, 255)
