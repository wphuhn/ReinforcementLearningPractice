from rl_functions.utilities import (
    run_summary,
    step_statistics,
)

def test_run_summary_outputs_properly_formatted_string():
    expected = "Episode: 2 , Elapsed Time for Episode: 0.001 s , Num. Steps in Episode: 500 , Cumulative Episode Reward: 9001"
    actual = run_summary(0.001, 2, 500, 9001)
    assert actual == expected

def test_step_statistics_outputs_properly_formatted_string():
    expected = "    Step: 3 , Reward This Step: 17 , Cumulative Reward This Episode: 9002 , Info: 9"
    actual = step_statistics(3, 17, 9002, 9)
    assert actual == expected
