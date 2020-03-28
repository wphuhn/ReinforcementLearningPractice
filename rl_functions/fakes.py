"""Collection of fake objects used in the test suite.
"""

class FakeEnv(object):
    """Fake object for modeling an OpenAI Gym environment in the test suite.
    """
    def __init__(self, observation, reward, done, info):
        """Initializes the fake object's return parameters.

        All parameters may independently be specified either as a scalar or as a
        list, in which case the relevant return value will cycle through the
        list each time step() is called.
        """
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.n_step = 0

    def step(self, action):
        """Mimics the step() subroutine for an OpenAI Gym environment.

        The action parameter here exists to keep the API consistent, it is not
        used.
        """
        observation = self._current_value(self.observation, self.n_step)
        reward = self._current_value(self.reward, self.n_step)
        done = self._current_value(self.done, self.n_step)
        info = self._current_value(self.info, self.n_step)
        self.n_step += 1
        return observation, reward, done, info

    @staticmethod
    def _current_value(value, step):
        """Determines which value to return based on the current step number.

        If the value is a scalar, it will always return that value.  If it is a
        list, it will cycle through the list.
        """
        if isinstance(value, list):
            index = step % len(value)
            return value[index]
        return value
