import numpy as np
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from env import ArmEnv  # Make sure env.py is in the same directory

class ArmPyEnvWrapper(py_environment.PyEnvironment):
    def __init__(self, image_size=(64, 64)):
        super().__init__()
        self._env = ArmEnv()
        self._image_size = image_size
        self.dt = 0.1  # refresh rate
        self.current_step = 0

        # Define action spec: 2D continuous action between -1 and 1
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action'
        )

        self._observation_spec = {
            'pixels': array_spec.BoundedArraySpec(
                shape=(image_size[0], image_size[1], 3),
                dtype=np.float32,
                minimum=0.0,
                maximum=1.0,
                name='pixels'
            )
        }

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._env.reset()
        obs = self._env.get_image_observation(resize=self._image_size, as_tensor=False)
        self._episode_ended = False
        print("[RESET DEBUG] obs shape:", obs.shape, "max:", np.max(obs), "mean:", np.mean(obs))
        return ts.restart({'pixels': obs})

    def _step(self, action, current_step=0):
        #add a counter to check the number of steps
        if self._episode_ended:
            return self._reset()
        
        self.current_step += 1
        if self.current_step > 20:
            print("[STEP DEBUG] Max steps reached, resetting environment.")
            return self._reset()

        _, reward, done = self._env.step(action)
        obs = self._env.get_image_observation(resize=self._image_size, as_tensor=False)
        self._episode_ended = done
        # print(f"[STEP DEBUG] Action taken: {action}, Reward: {reward}, Done: {done}")

        if done:
            return ts.termination({'pixels': obs}, reward)
        else:
            return ts.transition({'pixels': obs}, reward=reward, discount=1.0)