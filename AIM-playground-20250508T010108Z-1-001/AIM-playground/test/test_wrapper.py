from arm_env_wrapper import ArmPyEnvWrapper
import numpy as np

env = ArmPyEnvWrapper()
time_step = env.reset()
print("Initial observation shape:", time_step.observation.shape)

for _ in range(5):
    action = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
    time_step = env.step(action)
    print("Step observation shape:", time_step.observation.shape)
    print("Reward:", time_step.reward)
    print("Done:", time_step.is_last())
