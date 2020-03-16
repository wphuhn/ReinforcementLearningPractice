import numpy as np
import pytest

from rl_functions.utilities import (
    StateEncoder,
)

def test_state_encoder_throws_exception_when_grid_and_min_values_have_different_dimensions():
    grid = [2, 2]
    min_values = [0]
    max_values = [0, 1]
    with pytest.raises(Exception) as excinfo:
        _ = StateEncoder().fit(grid, min_values, max_values)
    assert "grid and minimum values arrays have different dimensions" in str(excinfo.value)

def test_state_encoder_throws_exception_when_grid_and_max_values_have_different_dimensions():
    grid = [3, 7, 5]
    min_values = [-1, -2, 4]
    max_values = [0, 5]
    with pytest.raises(Exception) as excinfo:
        _ = StateEncoder().fit(grid, min_values, max_values)
    assert "grid and maximum values arrays have different dimensions" in str(excinfo.value)

def test_state_encoder_throws_exception_when_min_value_greater_than_max_value():
    grid = [3, 7, 5]
    min_values = [-1, -2, 4]
    max_values = [0, 5, 3]
    with pytest.raises(Exception) as excinfo:
        _ = StateEncoder().fit(grid, min_values, max_values)
    assert "Minimum value 4 is greater than maximum value 3 for dimension 2" in str(excinfo.value)

def test_state_encoder_throws_exception_when_min_value_equal_max_value():
    grid = [3, 7, 5]
    min_values = [-1, -2, 4]
    max_values = [0, 5, 4]
    with pytest.raises(Exception) as excinfo:
        _ = StateEncoder().fit(grid, min_values, max_values)
    assert "Minimum value 4 is equal to maximum value 4 for dimension 2; to omit this dimension, use grid value of 1 instead" in str(excinfo.value)

def test_state_encoder_throws_exception_when_dimension_of_coords_does_not_equal_dimension_of_grid():
    grid = [3, 7, 5]
    min_values = [-1, -2, 2]
    max_values = [0, 5, 4]
    coords = [-0.019, 4.213]
    encoder = StateEncoder().fit(grid, min_values, max_values)
    with pytest.raises(Exception) as excinfo:
        _ = encoder.transform(coords)
    assert "Dimension of real-space coordinates differs from the encoder's grid" in str(excinfo.value)

@pytest.mark.parametrize("coords,expected", [
    ([-3], 0),
    ([-2], 0),
    ([-1.5], 0),
    ([-1], 1),
    ([-0.5], 1),
    ([0], 2),
    ([0.5], 2),
    ([1], 3),
    ([1.5], 3),
    ([2], 3),
    ([3], 3)
])
def test_state_encoder_gives_expected_results_when_1D_encoder_set_up(coords, expected):
    grid = [4]
    min_values = [-2]
    max_values = [2]
    encoder = StateEncoder().fit(grid, min_values, max_values)
    actual = encoder.transform(coords)
    assert expected == actual

@pytest.mark.parametrize("coords,expected", [
    ([-0.5, 0.5], 0),
    ([   0, 0.5], 0),
    ([ 0.5, 0.5], 0),
    ([   1, 0.5], 1),
    ([ 1.5, 0.5], 1),
    ([   2, 0.5], 1),
    ([ 2.5, 0.5], 1),
    ([-0.5,   1], 0),
    ([   0,   1], 0),
    ([ 0.5,   1], 0),
    ([   1,   1], 1),
    ([ 1.5,   1], 1),
    ([   2,   1], 1),
    ([ 2.5,   1], 1),
    ([-0.5, 1.5], 0),
    ([   0, 1.5], 0),
    ([ 0.5, 1.5], 0),
    ([   1, 1.5], 1),
    ([ 1.5, 1.5], 1),
    ([   2, 1.5], 1),
    ([ 2.5, 1.5], 1),
    ([-0.5,   2], 2),
    ([   0,   2], 2),
    ([ 0.5,   2], 2),
    ([   1,   2], 3),
    ([ 1.5,   2], 3),
    ([   2,   2], 3),
    ([ 2.5,   2], 3),
    ([-0.5, 2.5], 2),
    ([   0, 2.5], 2),
    ([ 0.5, 2.5], 2),
    ([   1, 2.5], 3),
    ([ 1.5, 2.5], 3),
    ([   2, 2.5], 3),
    ([ 2.5, 2.5], 3),
    ([-0.5,   3], 2),
    ([   0,   3], 2),
    ([ 0.5,   3], 2),
    ([   1,   3], 3),
    ([ 1.5,   3], 3),
    ([   2,   3], 3),
    ([ 2.5,   3], 3),
    ([-0.5, 3.5], 2),
    ([   0, 3.5], 2),
    ([ 0.5, 3.5], 2),
    ([   1, 3.5], 3),
    ([ 1.5, 3.5], 3),
    ([   2, 3.5], 3),
    ([ 2.5, 3.5], 3),
])
def test_state_encoder_gives_expected_results_when_2D_encoder_set_up(coords, expected):
    grid = [2, 2]
    min_values = [0, 1]
    max_values = [2, 3]
    encoder = StateEncoder().fit(grid, min_values, max_values)
    actual = encoder.transform(coords)
    assert expected == actual

# You can proably guess why I'm not doing 3D+ tests.

def test_2D_state_encoder_ignores_dimension_when_its_grid_interval_set_to_one():
    grid_1D = [2]
    min_values_1D = [0]
    max_values_1D = [2]
    grid_2D = [2, 1]
    min_values_2D = [0, 1]
    max_values_2D = [2, 3]
    encoder_1D = StateEncoder().fit(grid_1D, min_values_1D, max_values_1D)
    encoder_2D = StateEncoder().fit(grid_2D, min_values_2D, max_values_2D)
    for x in np.linspace(-1, 3, 10):
         expected = encoder_1D.transform([x])
         for y in np.linspace(0, 4, 10):
             actual = encoder_2D.transform([x, y])
             assert expected == actual
