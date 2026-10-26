from car_sim import config


def test_config_matches_spec():
    assert len(config.SENSOR_ANGLES) == 5
    assert config.STATE_DIM == 8
    assert config.ACTION_DIM == len(config.STEER_MAP) == 5
    assert config.R_CRASH < 0
    assert config.R_STALL < 0
    assert config.EPS_END < config.EPS_START


def test_map_and_defaults():
    assert config.GAME_W == 1400
    assert config.GAME_H == 800
    assert (config.START_X, config.START_Y) == (100, 400)
    assert (config.GOAL_X, config.GOAL_Y) == (1300, 400)
    assert config.GOAL_RADIUS == 30
    assert config.RANDOM_SPAWN is False
