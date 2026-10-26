import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_maps_dir(tmp_path, monkeypatch):
    """Tests never read or write the real maps/ folder (a saved default.json would change them)."""
    from car_sim import config

    monkeypatch.setattr(config, "MAPS_DIR", str(tmp_path / "maps"))
