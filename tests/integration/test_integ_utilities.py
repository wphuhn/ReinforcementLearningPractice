from rl_functions.utilities import (
    make_envs,
    close_envs,
)

from constants import ENV_NAME

#TODO: Tests for make_envs

def test_close_envs_closes_only_main_env_when_not_outputting_movie():
    env, env_raw, _ = make_envs(ENV_NAME, output_movie=False)
    close_envs(env)
    assert env is not None
    # This is what the close functionality does for Atari environments
    assert env.viewer is None

def test_close_envs_closes_both_env_when_outputting_movie():
    env, env_raw, _ = make_envs(ENV_NAME, output_movie=True)
    close_envs(env, env_raw)
    assert env is not None
    assert env_raw is not None
    # This is what the close functionality does for Atari environments
    assert env.viewer is None
    assert env_raw.viewer is None
