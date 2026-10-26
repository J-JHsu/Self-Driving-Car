"""Shortest route around the walls to the finish, on a coarse grid.

The grid is flood-filled (Dijkstra) outward from the finish, so every free cell
knows its route distance and which neighbor is the next step toward the finish.
The AI is told that direction and distance; it still does the driving itself.
"""
import heapq
import math

import numpy as np

from car_sim import config

_SQRT2 = math.sqrt(2.0)
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class RouteField:
    def __init__(self):
        self.builds = 0
        self._rows = config.GAME_H // config.GRID_CELL
        self._cols = config.GAME_W // config.GRID_CELL
        self._key = None            # what the current field was built for
        self._world = None          # held so its id cannot be reused by another world
        self._dist = None           # (rows, cols) float, inf where unreachable
        self._blocked = None        # (rows, cols) bool
        self._next = {}             # (r, c) -> (r, c) one step closer to the finish
        self._goal = (config.GOAL_X, config.GOAL_Y)
        self._start_ok = False

    # --- building ----------------------------------------------------------
    def ensure(self, world):
        """Rebuild if the walls, start or finish changed. True if the start has a route."""
        key = (world.version, tuple(world.goal))
        if world is not self._world or key != self._key:
            self._build(world)
            self._world = world
            self._key = key
        return self._start_ok

    def _build(self, world):
        cell = config.GRID_CELL
        rows, cols = self._rows, self._cols
        wall_dist = self._wall_distance(world)
        blocked = wall_dist < config.WALL_CLEARANCE
        self._goal = (float(world.goal[0]), float(world.goal[1]))
        gr, gc = self._cell_of(*self._goal)
        blocked[gr, gc] = False  # the finish cell is always free

        dist = np.full((rows, cols), math.inf)
        blk = blocked.tolist()
        # Cells where the car would not crash: the finish circle may reach closer to a wall
        # than the route itself is allowed to run.
        crash_free = (wall_dist >= config.CAR_W / 2).tolist()
        # Extra cost per pixel for cells close to a wall (0 beyond WALL_SOFT_ZONE).
        soft = np.clip((config.WALL_SOFT_ZONE - wall_dist) /
                       (config.WALL_SOFT_ZONE - config.WALL_CLEARANCE), 0.0, 1.0) * config.WALL_SOFT_COST
        weight = (1.0 + soft).tolist()
        d = [[math.inf] * cols for _ in range(rows)]
        nxt = {}
        # The car wins anywhere within GOAL_RADIUS of the finish, and it can get that close to
        # a wall without crashing, so every crash-free cell inside that circle is a starting
        # point. A finish touching a wall or the border is then still reachable.
        heap = []
        seeds = set()
        reach = config.GOAL_RADIUS + cell
        for r in range(max(0, gr - int(reach // cell) - 1), min(rows, gr + int(reach // cell) + 2)):
            for c in range(max(0, gc - int(reach // cell) - 1), min(cols, gc + int(reach // cell) + 2)):
                cx, cy = (c + 0.5) * cell, (r + 0.5) * cell
                gap = math.hypot(cx - self._goal[0], cy - self._goal[1])
                if (r, c) == (gr, gc) or (crash_free[r][c] and gap <= config.GOAL_RADIUS):
                    d[r][c] = 0.0 if (r, c) == (gr, gc) else gap
                    heap.append((d[r][c], r, c))
                    seeds.add((r, c))
        heapq.heapify(heap)
        while heap:
            cur, r, c = heapq.heappop(heap)
            if cur > d[r][c]:
                continue
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or blk[nr][nc]:
                    continue
                if dr and dc and blk[r + dr][c] and blk[r][c + dc] and (r, c) not in seeds:
                    continue  # no squeezing diagonally between two walls
                nd = cur + (_SQRT2 if dr and dc else 1.0) * cell * 0.5 * (weight[r][c] + weight[nr][nc])
                if nd < d[nr][nc]:
                    d[nr][nc] = nd
                    nxt[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (nd, nr, nc))
        dist[:] = np.array(d)
        self._dist, self._blocked, self._next = dist, blocked, nxt
        self.builds += 1
        sr, sc = self._free_cell(*world.start)
        self._start_ok = sr is not None and math.isfinite(dist[sr, sc])

    def _wall_distance(self, world):
        """Distance from each cell center to the nearest wall or map border, capped at the soft zone."""
        cell = config.GRID_CELL
        cap = config.WALL_SOFT_ZONE
        rows, cols = self._rows, self._cols
        xs = (np.arange(cols) + 0.5) * cell
        ys = (np.arange(rows) + 0.5) * cell
        px, py = np.meshgrid(xs, ys)
        near = np.minimum(np.minimum(px, config.GAME_W - px), np.minimum(py, config.GAME_H - py))
        dist = np.minimum(near, float(cap))
        # Only look at the box around each segment, widened by the cap.
        for (ax, ay), (bx, by) in world.obstacles:
            c0 = max(0, int((min(ax, bx) - cap) // cell))
            c1 = min(cols - 1, int((max(ax, bx) + cap) // cell))
            r0 = max(0, int((min(ay, by) - cap) // cell))
            r1 = min(rows - 1, int((max(ay, by) + cap) // cell))
            if c0 > c1 or r0 > r1:
                continue
            qx, qy = px[r0:r1 + 1, c0:c1 + 1], py[r0:r1 + 1, c0:c1 + 1]
            abx, aby = bx - ax, by - ay
            len2 = abx * abx + aby * aby
            if len2 == 0:
                t = np.zeros_like(qx)
            else:
                t = np.clip(((qx - ax) * abx + (qy - ay) * aby) / len2, 0.0, 1.0)
            d = np.hypot(qx - (ax + t * abx), qy - (ay + t * aby))
            sub_ = dist[r0:r1 + 1, c0:c1 + 1]
            np.minimum(sub_, d, out=sub_)
        return dist

    # --- lookups -----------------------------------------------------------
    def _cell_of(self, x, y):
        c = min(self._cols - 1, max(0, int(x // config.GRID_CELL)))
        r = min(self._rows - 1, max(0, int(y // config.GRID_CELL)))
        return r, c

    def _free_cell(self, x, y):
        """The query's cell, or the nearest free cell within 3 cells, else (None, None)."""
        if self._dist is None:
            return None, None
        r, c = self._cell_of(x, y)
        if not self._blocked[r, c]:
            return r, c
        best, best_d = None, math.inf
        cell = config.GRID_CELL
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._rows and 0 <= nc < self._cols and not self._blocked[nr, nc]:
                    dd = math.hypot((nc + 0.5) * cell - x, (nr + 0.5) * cell - y)
                    if dd < best_d:
                        best, best_d = (nr, nc), dd
        return best if best else (None, None)

    def _center(self, r, c):
        return (c + 0.5) * config.GRID_CELL, (r + 0.5) * config.GRID_CELL

    def reachable(self, x, y):
        r, c = self._free_cell(x, y)
        return r is not None and math.isfinite(self._dist[r, c])

    def direction(self, x, y):
        """Absolute angle of the next step along the route, or None if there is no route."""
        r, c = self._free_cell(x, y)
        if r is None or not math.isfinite(self._dist[r, c]):
            return None
        cx, cy = self._center(r, c)
        step = self._next.get((r, c))
        if step is None:  # the finish cell itself: head for the finish
            tx, ty = self._goal
            if math.hypot(tx - x, ty - y) < 1e-6:
                return 0.0
            return math.atan2(ty - y, tx - x)
        tx, ty = self._center(*step)
        return math.atan2(ty - cy, tx - cx)

    def distance(self, x, y):
        """Route length in pixels to the finish; math.inf if there is none."""
        r, c = self._free_cell(x, y)
        if r is None or not math.isfinite(self._dist[r, c]):
            return math.inf
        cx, cy = self._center(r, c)
        step = self._next.get((r, c))
        if step is None:
            return math.hypot(self._goal[0] - x, self._goal[1] - y)
        # Slide along the step direction so the value changes smoothly inside a cell.
        sx, sy = self._center(*step)
        norm = math.hypot(sx - cx, sy - cy)
        along = ((x - cx) * (sx - cx) + (y - cy) * (sy - cy)) / norm
        along = max(-config.GRID_CELL, min(config.GRID_CELL, along))
        return max(0.0, float(self._dist[r, c]) - along)

    def path(self, x, y, max_points=400):
        """The route from (x, y) to the finish as a list of points (empty if no route)."""
        r, c = self._free_cell(x, y)
        if r is None or not math.isfinite(self._dist[r, c]):
            return []
        pts = [(float(x), float(y))]
        while (r, c) in self._next:
            r, c = self._next[(r, c)]
            pts.append(self._center(r, c))
        pts.append(self._goal)
        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
            pts = [pts[i] for i in idx]
        return pts


class StraightField:
    """The 'no route' alternative: only the straight line to the finish, ignoring walls.

    Same interface as RouteField, so the environment can use either. The car knows where
    the finish is but not how to get around anything, so it has to learn that by crashing.
    """

    def __init__(self):
        self.builds = 0  # bumps when the finish moves, so the env does not count it as progress
        self._goal = None

    def ensure(self, world):
        goal = tuple(world.goal)
        if goal != self._goal:
            self._goal = goal
            self.builds += 1
        return True

    def reachable(self, x, y):
        return True

    def distance(self, x, y):
        return math.hypot(self._goal[0] - x, self._goal[1] - y)

    def direction(self, x, y):
        return math.atan2(self._goal[1] - y, self._goal[0] - x)

    def path(self, x, y, max_points=400):
        return [(float(x), float(y)), (float(self._goal[0]), float(self._goal[1]))]
