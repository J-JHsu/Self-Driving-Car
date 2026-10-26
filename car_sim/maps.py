"""Saving and loading maps (walls, start, finish) as JSON files in config.MAPS_DIR.

File layout: {"version": 1, "walls": [[x1, y1, x2, y2], ...], "start": [x, y], "goal": [x, y]}.
Nothing here raises for a missing or damaged file: the functions return a message for the
window to show, and a failed load leaves the world exactly as it was.
"""
import argparse
import json
import math
import os
import re

from car_sim import config

_NAME = re.compile(r"[\w-]+")


class MapError(Exception):
    """A map file that cannot be used. `reason` finishes the sentence '<path> ...'."""

    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason


def map_name_arg(text):
    """argparse type for --map: a plain file name, so it cannot point outside maps/."""
    if not _NAME.fullmatch(text) or len(text) > config.MAP_NAME_MAX:
        raise argparse.ArgumentTypeError(
            f"map names use letters, digits, _ and - only (at most {config.MAP_NAME_MAX} characters)"
        )
    return text


def map_path(name):
    return os.path.join(config.MAPS_DIR, f"{name}.json")


def display_path(name):
    """The path as shown to the user, with forward slashes on every platform."""
    return f"{config.MAPS_DIR}/{name}.json"


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value, w, h, strict):
    """A [x, y] pair inside the map. Walls may touch the far edge; start and finish may not."""
    if not (isinstance(value, list) and len(value) == 2 and all(_number(v) for v in value)):
        raise MapError("is damaged", "a point is not two numbers")
    x, y = value
    inside = (0 <= x < w and 0 <= y < h) if strict else (0 <= x <= w and 0 <= y <= h)
    if not inside:
        raise MapError("is damaged", "a point is outside the map")
    return (x, y)


def _parse(data):
    """Validate decoded JSON and return (walls, start, goal); raises MapError."""
    if not isinstance(data, dict):
        raise MapError("is damaged", "the top level is not an object")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise MapError("is damaged", "no version number")
    if version > config.MAP_FORMAT_VERSION:
        raise MapError("is from a newer version", f"map version {version}")
    walls = data.get("walls")
    if not isinstance(walls, list):
        raise MapError("is damaged", "walls is not a list")
    w, h = config.GAME_W, config.GAME_H
    parsed = []
    for wall in walls:
        if not (isinstance(wall, list) and len(wall) == 4):
            raise MapError("is damaged", "a wall is not four numbers")
        parsed.append((_point(wall[:2], w, h, False), _point(wall[2:], w, h, False)))
    return parsed, _point(data.get("start"), w, h, True), _point(data.get("goal"), w, h, True)


def read_map(world, path):
    """Load the file at `path` into `world`. Raises MapError and leaves the world untouched."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise MapError("was not found", str(exc)) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise MapError("is damaged", f"{type(exc).__name__}: {exc}") from exc
    except OSError as exc:  # a folder instead of a file, no permission, ...
        raise MapError("could not be read", str(exc)) from exc
    walls, start, goal = _parse(data)
    world.replace(walls, start, goal)


def load_startup(world, name=None):
    """Load map `name` (default: config.DEFAULT_MAP) into the new `world`.

    A missing default map is normal and silent; a missing map that was asked for by name,
    or any damaged one, gives a message and the world stays as it was. Returns the message
    to show in the window (or None) and prints it too.
    """
    explicit = name is not None
    name = name or config.DEFAULT_MAP
    shown = display_path(name)
    try:
        read_map(world, map_path(name))
    except MapError as exc:
        if exc.reason == "was not found" and not explicit:
            return None
        message = f"{shown} {exc.reason} - starting with an empty map"
        print(f"{message} ({exc})" if str(exc) != exc.reason else message)
        return message
    message = f"Loaded {shown}"
    print(message)
    return message


def save_map(world, name=None):
    """Write the world to maps/NAME.json (creating maps/). Returns the message to show."""
    name = name or config.DEFAULT_MAP
    path = map_path(name)
    data = {
        "version": config.MAP_FORMAT_VERSION,
        "walls": [[a[0], a[1], b[0], b[1]] for a, b in world.obstacles],
        "start": list(world.start),
        "goal": list(world.goal),
    }
    tmp = path + ".tmp"
    try:
        os.makedirs(config.MAPS_DIR, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # an interrupted save never leaves a half-written map
    except OSError as exc:
        print(f"Could not save {display_path(name)}: {exc}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return f"Could not save {display_path(name)}"
    message = f"Saved {display_path(name)}"
    print(message)
    return message
