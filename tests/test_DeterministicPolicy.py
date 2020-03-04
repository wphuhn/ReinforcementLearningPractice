import pytest

from rl_functions.policies import DeterministicPolicy

@pytest.fixture
def det_policy():
    return DeterministicPolicy()

def test_object_instantiation_is_successful_when_no_parameters_are_supplied(det_policy):
    assert isinstance(det_policy, DeterministicPolicy)

def test_policy_throws_exception_when_registering_transition_with_negative_state_index(det_policy):
    invalid_state = -1
    valid_action = 25
    invalid_dict = {invalid_state: valid_action}
    with pytest.raises(Exception) as excinfo:
        det_policy.add_transitions(invalid_dict)
    assert "State index in deterministic policy must be non-negative" in str(excinfo.value)

def test_policy_throws_exception_when_registering_transition_with_negative_action_index(det_policy):
    valid_state = 50
    invalid_action = -1
    invalid_dict = {valid_state: invalid_action}
    with pytest.raises(Exception) as excinfo:
        det_policy.add_transitions(invalid_dict)
    assert "Action index in deterministic policy must be non-negative" in str(excinfo.value)

def test_policy_throws_exception_when_predicting_next_action_for_state_that_has_not_been_registered(det_policy):
    state_with_no_transition = 50
    with pytest.raises(Exception) as excinfo:
        _ = det_policy.next_action(state_with_no_transition)
    assert "State 50 in deterministic policy has no action registered with it" in str(excinfo.value)

def test_policy_provides_next_action_when_state_with_valid_state_action_pair_has_been_registered(det_policy):
    state = 50
    expected = 25
    transitions = {state: expected}
    det_policy.add_transitions(transitions)
    actual = det_policy.next_action(state)
    assert expected == actual

def test_policy_can_be_generated_greedily_from_q_function_when_unique_maximum_exist_for_every_state(det_policy):
    state_1 = 5
    state_2 = 15
    expected_1 = 3
    expected_2 = 1
    q = {
        state_1: {
            0: 15.,
            1: 30.,
            2: 1.,
            expected_1: 40.,
        },
        state_2: {
            0: 30.,
            expected_2: 100.,
            2: -15.,
            3: 7.,
            4: 12.,
        },
    }
    det_policy.generate_greedily_from_q(q)
    actual_1 = det_policy.next_action(state_1)
    actual_2 = det_policy.next_action(state_2)
    assert expected_1 == actual_1
    assert expected_2 == actual_2

def test_policy_throws_exception_when_attempting_to_generate_greedily_from_q_function_but_more_than_one_maximum_exists(det_policy):
    q = {
        3: {
            0: 15.,
            1: 30.,
            2: 1.,
            3: 30.,
        },
    }
    with pytest.raises(Exception) as excinfo:
        det_policy.generate_greedily_from_q(q)
    assert "state 3 does not have a unique greedy action in q function, cannot generate a deterministic policy for it" in str(excinfo.value)
