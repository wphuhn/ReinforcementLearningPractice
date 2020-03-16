from rl_functions.utilities import (
    run_summary,
    step_statistics,
    make_envs,
    close_envs,
)

from constants import ENV_NAME

def test_run_summary_outputs_properly_formatted_string():
    expected = "Episode: 2 , Elapsed Time for Episode: 0.001 s , Num. Steps in Episode: 500 , Cumulative Episode Reward: 9001"
    actual = run_summary(0.001, 2, 500, 9001)
    assert actual == expected

def test_step_statistics_outputs_properly_formatted_string():
    expected = "    Step: 3 , Reward This Step: 17 , Cumulative Reward This Episode: 9002 , Info: 9"
    actual = step_statistics(3, 17, 9002, 9)
    assert actual == expected

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
