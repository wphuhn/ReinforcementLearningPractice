import pytest

from rl_functions.policies import DeterministicPolicy

def test_object_instantiation_is_successful_when_number_of_states_and_actions_are_supplied():
    n_states = 100
    n_actions = 50
    det_policy = DeterministicPolicy(n_states, n_actions)
    assert isinstance(det_policy, DeterministicPolicy)

def test_policy_throws_exception_when_registering_transition_with_negative_state_index():
    n_states = 100
    n_actions = 50
    invalid_state = -1
    valid_action = 25
    invalid_dict = {invalid_state: valid_action}
    det_policy = DeterministicPolicy(n_states, n_actions)
    with pytest.raises(Exception) as excinfo:
        det_policy.add_transitions(invalid_dict)
    assert "State index in deterministic policy must be non-negative" in str(excinfo.value)

def test_policy_throws_exception_when_registering_transition_with_negative_action_index():
    n_states = 100
    n_actions = 50
    valid_state = 50
    invalid_action = -1
    invalid_dict = {valid_state: invalid_action}
    det_policy = DeterministicPolicy(n_states, n_actions)
    with pytest.raises(Exception) as excinfo:
        det_policy.add_transitions(invalid_dict)
    assert "Action index in deterministic policy must be non-negative" in str(excinfo.value)

def test_policy_throws_exception_when_registering_transition_with_too_high_action_index():
    n_states = 100
    n_actions = 50
    invalid_state = 100
    valid_action = 25
    invalid_dict = {invalid_state: valid_action}
    det_policy = DeterministicPolicy(n_states, n_actions)
    with pytest.raises(Exception) as excinfo:
        det_policy.add_transitions(invalid_dict)
    assert "State index 100 in deterministic policy is larger than the maximum state index of 99" in str(excinfo.value)

def test_policy_throws_exception_when_registering_transition_with_too_high_action_index():
    n_states = 100
    n_actions = 50
    valid_state = 50
    invalid_action = 50
    invalid_dict = {valid_state: invalid_action}
    det_policy = DeterministicPolicy(n_states, n_actions)
    with pytest.raises(Exception) as excinfo:
        det_policy.add_transitions(invalid_dict)
    assert "Action index 50 in deterministic policy is larger than the maximum action index of 49" in str(excinfo.value)

def test_policy_throws_exception_when_predicting_next_action_for_state_that_has_not_been_registered():
    n_states = 100
    n_actions = 50
    state_with_no_transition = 50
    det_policy = DeterministicPolicy(n_states, n_actions)
    with pytest.raises(Exception) as excinfo:
        _ = det_policy.next_action(state_with_no_transition)
    assert "State 50 in deterministic policy has no action registered with it" in str(excinfo.value)

def test_policy_provides_next_action_when_state_with_valid_state_action_pair_has_been_registered():
    n_states = 100
    n_actions = 50
    state = 50
    expected = 25
    transitions = {state: expected}
    det_policy = DeterministicPolicy(n_states, n_actions)
    det_policy.add_transitions(transitions)
    actual = det_policy.next_action(state)
    assert expected == actual
