import json
import os
from collections import defaultdict

import pygame
import pytest

import car_hum
import self_driving_car_ai as ai
from car_sim import config, maps, ui
from car_sim.route import RouteField
from car_sim.world import World


def sample_world():
    world = World()
    world.add_segment((100, 100), (300, 120))
    world.add_segment((300, 120), (310.5, 400.25))
    world.add_segment((500, 500), (500, 500))  # a single click leaves a dot
    world.set_start((50, 60))
    world.set_goal((1200, 700))
    return world


def walled_off_world():
    world = World()
    world.add_segment((1200, 0), (1200, config.GAME_H))  # nothing can reach the finish
    return world


def write_map(name, world):
    maps.save_map(world, name)
    return maps.map_path(name)


# --- names and paths ---------------------------------------------------------

def test_map_path_is_maps_folder_name_json():
    assert maps.map_path("default") == os.path.join(config.MAPS_DIR, "default.json")
    assert maps.map_path("maze") == os.path.join(config.MAPS_DIR, "maze.json")


def test_default_map_name():
    assert config.DEFAULT_MAP == "default"
    assert maps.display_path("default") == f"{config.MAPS_DIR}/default.json"


@pytest.mark.parametrize("name", ["maze", "level_2", "my-map", "A1", "x" * config.MAP_NAME_MAX])
def test_good_map_names(name):
    assert maps.map_name_arg(name) == name


@pytest.mark.parametrize("name", ["", "../secret", "a/b", "a\\b", "x.json", "..", ".", "a b", "x" * (config.MAP_NAME_MAX + 1)])
def test_bad_map_names_are_rejected(name):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        maps.map_name_arg(name)


# --- saving and loading ------------------------------------------------------

def test_saved_map_loads_back_identical():
    original = sample_world()
    maps.save_map(original, "round")
    loaded = World()
    maps.read_map(loaded, maps.map_path("round"))
    assert loaded.obstacles == original.obstacles
    assert loaded.start == original.start
    assert loaded.goal == original.goal


def test_save_creates_the_folder_and_reports_it():
    assert not os.path.exists(config.MAPS_DIR)
    message = maps.save_map(sample_world(), "default")
    assert message == f"Saved {config.MAPS_DIR}/default.json"
    assert os.path.isfile(maps.map_path("default"))


def test_file_format_has_version_walls_start_finish():
    maps.save_map(sample_world(), "fmt")
    with open(maps.map_path("fmt"), encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == config.MAP_FORMAT_VERSION
    assert data["walls"][0] == [100, 100, 300, 120]
    assert data["start"] == [50, 60] and data["goal"] == [1200, 700]


def test_save_replaces_an_existing_file_and_leaves_no_temp_file():
    maps.save_map(sample_world(), "twice")
    maps.save_map(World(), "twice")
    loaded = sample_world()
    maps.read_map(loaded, maps.map_path("twice"))
    assert loaded.obstacles == []
    assert os.listdir(config.MAPS_DIR) == ["twice.json"]


def test_save_failure_is_reported_not_raised(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    monkeypatch.setattr(config, "MAPS_DIR", str(blocker))  # a folder cannot be created here
    message = maps.save_map(sample_world(), "default")
    assert message.startswith("Could not save")


def test_load_bumps_the_version_and_the_route_rebuilds():
    maps.save_map(walled_off_world(), "wall")
    world = World()
    route = RouteField()
    assert route.ensure(world) is True
    builds, version = route.builds, world.version

    maps.read_map(world, maps.map_path("wall"))

    assert world.version > version
    assert route.ensure(world) is False  # rebuilt: the loaded wall blocks the finish
    assert route.builds == builds + 1


def test_loading_the_same_map_twice_still_bumps_the_version():
    maps.save_map(sample_world(), "same")
    world = World()
    maps.read_map(world, maps.map_path("same"))
    first = world.version
    maps.read_map(world, maps.map_path("same"))
    assert world.version > first


def test_load_replaces_the_old_walls_and_the_sensor_cache():
    world = sample_world()
    world.cast_ray(400, 400, 0.0)  # fills the numpy cache
    maps.save_map(World(), "empty")
    maps.read_map(world, maps.map_path("empty"))
    assert world.obstacles == []
    assert world.cast_ray(100, 110, 0.0) == config.MAX_SENSOR_DISTANCE


# --- missing and damaged files -----------------------------------------------

def test_missing_default_map_is_silent_and_leaves_the_empty_map(capsys):
    world = World()
    assert maps.load_startup(world, None) is None
    assert world.obstacles == [] and world.start == (config.START_X, config.START_Y)
    assert capsys.readouterr().out == ""


def test_missing_named_map_shows_a_message_and_starts_empty(capsys):
    world = World()
    message = maps.load_startup(world, "nope")
    assert message == f"{config.MAPS_DIR}/nope.json was not found - starting with an empty map"
    assert world.obstacles == []
    assert message in capsys.readouterr().out


def test_good_map_loads_at_startup_with_a_message():
    maps.save_map(sample_world(), "default")
    world = World()
    message = maps.load_startup(world, None)
    assert message == f"Loaded {config.MAPS_DIR}/default.json"
    assert world.obstacles == sample_world().obstacles


DAMAGED = {
    "not json": "this is not json",
    "empty file": "",
    "truncated": '{"version": 1, "walls": [[1, 2, 3',
    "top level list": "[1, 2, 3]",
    "no version": json.dumps({"walls": [], "start": [1, 1], "goal": [2, 2]}),
    "version is text": json.dumps({"version": "1", "walls": [], "start": [1, 1], "goal": [2, 2]}),
    "walls missing": json.dumps({"version": 1, "start": [1, 1], "goal": [2, 2]}),
    "walls not a list": json.dumps({"version": 1, "walls": 5, "start": [1, 1], "goal": [2, 2]}),
    "wall with 3 numbers": json.dumps({"version": 1, "walls": [[1, 2, 3]], "start": [1, 1], "goal": [2, 2]}),
    "wall with text": json.dumps({"version": 1, "walls": [[1, 2, 3, "x"]], "start": [1, 1], "goal": [2, 2]}),
    "wall with true": json.dumps({"version": 1, "walls": [[1, 2, 3, True]], "start": [1, 1], "goal": [2, 2]}),
    "wall with nan": '{"version": 1, "walls": [[1, 2, 3, NaN]], "start": [1, 1], "goal": [2, 2]}',
    "wall off the map": json.dumps({"version": 1, "walls": [[1, 2, 3, 10**9]], "start": [1, 1], "goal": [2, 2]}),
    "start missing": json.dumps({"version": 1, "walls": [], "goal": [2, 2]}),
    "start one number": json.dumps({"version": 1, "walls": [], "start": [1], "goal": [2, 2]}),
    "start off the map": json.dumps({"version": 1, "walls": [], "start": [-5, 1], "goal": [2, 2]}),
    "goal off the map": json.dumps({"version": 1, "walls": [], "start": [1, 1], "goal": [config.GAME_W, 2]}),
    "too deeply nested": "[" * 100000,
}


@pytest.mark.parametrize("text", DAMAGED.values(), ids=DAMAGED.keys())
def test_damaged_file_is_reported_and_the_world_is_untouched(text, capsys):
    os.makedirs(config.MAPS_DIR)
    with open(maps.map_path("default"), "w", encoding="utf-8") as f:
        f.write(text)
    world = World()
    message = maps.load_startup(world, None)
    assert message == f"{config.MAPS_DIR}/default.json is damaged - starting with an empty map"
    assert world.obstacles == []
    assert (world.start, world.goal) == ((config.START_X, config.START_Y), (config.GOAL_X, config.GOAL_Y))
    assert world.version == 0
    assert "damaged" in capsys.readouterr().out


def test_binary_garbage_is_damaged_not_a_crash():
    os.makedirs(config.MAPS_DIR)
    with open(maps.map_path("default"), "wb") as f:
        f.write(b"\xff\xfe\x00\x80garbage")
    assert "damaged" in maps.load_startup(World(), None)


def test_a_folder_in_place_of_the_file_is_unreadable_not_a_crash():
    os.makedirs(maps.map_path("default"))
    message = maps.load_startup(World(), None)
    assert message == f"{config.MAPS_DIR}/default.json could not be read - starting with an empty map"


def test_map_from_a_newer_version_is_refused():
    os.makedirs(config.MAPS_DIR)
    data = {"version": config.MAP_FORMAT_VERSION + 1, "walls": [], "start": [1, 1], "goal": [2, 2]}
    with open(maps.map_path("default"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    message = maps.load_startup(World(), None)
    assert message == (f"{config.MAPS_DIR}/default.json is from a newer version - "
                       "starting with an empty map")


def test_longest_messages_fit_the_panel(monkeypatch):
    monkeypatch.setattr(config, "MAPS_DIR", "maps")  # the real folder name, not the test's temp dir
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    limit = ui.WHEEL_CENTER[0] - ui.WHEEL_RADIUS - 10
    name = "x" * config.MAP_NAME_MAX
    shown = maps.display_path(name)
    for line in (f"{shown} is from a newer version - starting with an empty map",
                 f"{shown} could not be read - starting with an empty map",
                 f"{shown} was not found - starting with an empty map",
                 f"Could not save {shown}"):
        assert font.size(line)[0] + 15 < limit, line


# --- keys and hints ----------------------------------------------------------

def key_event(key, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


def test_ctrl_s_is_the_save_key_and_plain_keys_are_not():
    assert ui.is_save_event(key_event(pygame.K_s, pygame.KMOD_LCTRL))
    assert ui.is_save_event(key_event(pygame.K_s, pygame.KMOD_RCTRL | pygame.KMOD_NUM))
    assert not ui.is_save_event(key_event(pygame.K_s))
    assert not ui.is_save_event(key_event(pygame.K_s, pygame.KMOD_SHIFT))
    assert not ui.is_save_event(key_event(pygame.K_a, pygame.KMOD_LCTRL))
    assert not ui.is_save_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_s, mod=pygame.KMOD_LCTRL))


def test_ctrl_s_does_not_clash_with_any_other_key():
    for hint in ui.ALL_HINTS:
        assert "Ctrl+S" not in hint or "save" in hint
    # The only other binding on S is the human brake, which must ignore it while Ctrl is held.
    keys = defaultdict(bool)
    keys[pygame.K_LCTRL] = True
    keys[pygame.K_s] = True
    assert car_hum.read_keys(keys) == (0.0, 0.0, 0.0)
    keys.clear()
    keys[pygame.K_RCTRL] = True
    keys[pygame.K_a] = True
    keys[pygame.K_w] = True
    assert car_hum.read_keys(keys) == (0.0, 0.0, 0.0)
    keys.clear()
    keys[pygame.K_s] = True  # braking alone still works
    assert car_hum.read_keys(keys) == (0.0, 1.0, 0.0)


def test_save_hint_is_on_screen_in_both_programs():
    assert "Ctrl+S: save map" in ui.HINT_VIEW
    assert "Ctrl+S: save map" in ui.HINT_VIEW_AI


# --- command line ------------------------------------------------------------

def test_ai_command_line_map_with_cars_and_no_route():
    assert ai.parse_args([]) == (True, config.NUM_CARS, None)
    assert ai.parse_args(["--map", "maze"]) == (True, config.NUM_CARS, "maze")
    assert ai.parse_args(["--no-route", "--cars", "3", "--map", "maze"]) == (False, 3, "maze")
    assert ai.parse_args(["--map", "maze", "--cars", "2"]) == (True, 2, "maze")


def test_ai_command_line_rejects_a_bad_map_name():
    with pytest.raises(SystemExit):
        ai.parse_args(["--map", "../x"])


def test_human_command_line_map():
    assert car_hum.parse_args([]) is None
    assert car_hum.parse_args(["--map", "maze"]) == "maze"
    with pytest.raises(SystemExit):
        car_hum.parse_args(["--map", "a/b"])


def test_map_flag_picks_the_file_in_training(tmp_path):
    write_map("default", World())
    write_map("maze", walled_off_world())
    model = str(tmp_path / "m.pt")

    plain = ai.train(headless=True, max_steps=30, model_path=model)
    assert plain["no_route"] is False and plain["steps"] == 30

    maze = ai.train(headless=True, max_steps=30, model_path=model, map_name="maze", num_cars=3)
    assert maze["no_route"] is True and maze["steps"] == 0  # the maze file was the one loaded


def test_map_flag_works_with_no_route_mode(tmp_path):
    write_map("maze", walled_off_world())
    result = ai.train(headless=True, max_steps=30, model_path=str(tmp_path / "m.pt"),
                      map_name="maze", num_cars=2, use_route=False)
    assert result["no_route"] is False and result["steps"] == 30


def test_training_starts_from_default_json_when_it_exists(tmp_path):
    write_map("default", walled_off_world())
    result = ai.train(headless=True, max_steps=30, model_path=str(tmp_path / "m.pt"))
    assert result["no_route"] is True


def test_training_survives_a_missing_or_damaged_map(tmp_path, capsys):
    model = str(tmp_path / "m.pt")
    result = ai.train(headless=True, max_steps=30, model_path=model, map_name="nope")
    assert result["steps"] == 30
    os.makedirs(config.MAPS_DIR, exist_ok=True)
    with open(maps.map_path("default"), "w") as f:
        f.write("{broken")
    result = ai.train(headless=True, max_steps=30, model_path=model)
    assert result["steps"] == 30 and result["no_route"] is False
    out = capsys.readouterr().out
    assert "was not found" in out and "is damaged" in out


def test_an_explicit_world_is_not_replaced_by_a_map_file(tmp_path):
    write_map("default", walled_off_world())
    result = ai.train(headless=True, max_steps=30, model_path=str(tmp_path / "m.pt"), world=World())
    assert result["no_route"] is False


# --- Ctrl+S inside the windows -----------------------------------------------

def scripted_events(monkeypatch, per_frame):
    """pygame.event.get returns per_frame[n] on frame n, and the real events otherwise."""
    frames = {"n": 0}
    real_get = pygame.event.get

    def scripted(*args, **kwargs):
        frames["n"] += 1
        return per_frame.get(frames["n"]) or real_get(*args, **kwargs)

    monkeypatch.setattr(pygame.event, "get", scripted)


def test_ctrl_s_in_the_ai_window_saves_to_the_named_map_and_shows_a_message(tmp_path, monkeypatch):
    world = sample_world()
    scripted_events(monkeypatch, {3: [key_event(pygame.K_s, pygame.KMOD_LCTRL)]})
    messages = []
    real_draw = ai._draw
    monkeypatch.setattr(ai, "_draw", lambda *a, **k: (messages.append(a[-1]), real_draw(*a, **k))[1])

    ai.train(headless=False, max_steps=40, model_path=str(tmp_path / "m.pt"), world=world,
             map_name="saved", num_cars=2)

    loaded = World()
    maps.read_map(loaded, maps.map_path("saved"))
    assert loaded.obstacles == world.obstacles
    assert (loaded.start, loaded.goal) == (world.start, world.goal)
    assert f"Saved {config.MAPS_DIR}/saved.json" in messages
    assert messages[0] == ""  # no message before anything happened


def test_ctrl_s_without_a_name_saves_default_json(tmp_path, monkeypatch):
    scripted_events(monkeypatch, {3: [key_event(pygame.K_s, pygame.KMOD_LCTRL)]})
    ai.train(headless=False, max_steps=10, model_path=str(tmp_path / "m.pt"), world=sample_world())
    assert os.path.isfile(maps.map_path("default"))


def test_plain_s_in_the_ai_window_saves_nothing(tmp_path, monkeypatch):
    scripted_events(monkeypatch, {3: [key_event(pygame.K_s)]})
    ai.train(headless=False, max_steps=10, model_path=str(tmp_path / "m.pt"), world=sample_world())
    assert not os.path.exists(config.MAPS_DIR)


def test_loaded_map_walls_appear_in_the_ai_window(tmp_path):
    write_map("default", sample_world())
    seen = {}
    real_draw = ai._draw

    def spy(screen, font, vec, *args):
        seen["walls"] = list(vec.world.obstacles)
        return real_draw(screen, font, vec, *args)

    ai._draw = spy
    try:
        ai.train(headless=False, max_steps=5, model_path=str(tmp_path / "m.pt"))
    finally:
        ai._draw = real_draw
    assert seen["walls"] == sample_world().obstacles


def test_ctrl_s_in_the_human_window_saves_and_loads_next_time(monkeypatch):
    scripted_events(monkeypatch, {
        2: [key_event(pygame.K_s, pygame.KMOD_LCTRL)],
        4: [pygame.event.Event(pygame.QUIT)],
    })
    car_hum.main(["--map", "human"])
    assert os.path.isfile(maps.map_path("human"))

    seen = {}
    real_init = car_hum.HumanGame.__init__

    def spy_init(self):
        real_init(self)
        seen["game"] = self

    monkeypatch.setattr(car_hum.HumanGame, "__init__", spy_init)
    write_map("human", sample_world())
    scripted_events(monkeypatch, {2: [pygame.event.Event(pygame.QUIT)]})
    car_hum.main(["--map", "human"])
    game = seen["game"]
    assert game.world.obstacles == sample_world().obstacles
    assert (game.car.x, game.car.y) == sample_world().start  # the car starts at the loaded start
