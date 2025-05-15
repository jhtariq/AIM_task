import numpy as np

def step(self, action):
    done = False
    action = np.clip(action, *self.action_bound)

    if not hasattr(self, 'prev_action'):
        self.prev_action = np.zeros_like(action)
    if not hasattr(self, 'cumulative_error'):
        self.cumulative_error = 0.0

    self.arm_info['r'] += action * self.dt
    self.arm_info['r'] %= np.pi * 2  # normalize

    (a1l, a2l) = self.arm_info['l']
    (a1r, a2r) = self.arm_info['r']
    a1xy = np.array([200., 200.])
    a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy
    finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_

    dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
    current_error = np.sqrt(dist2[0]**2 + dist2[1]**2)

    # PID-style reward
    p_term = -current_error
    d_term = -np.linalg.norm(action - self.prev_action)
    self.cumulative_error += current_error * self.dt
    i_term = -self.cumulative_error

    Kp, Ki, Kd = 1.0, 0.01, 0.1
    r = Kp * p_term + Ki * i_term + Kd * d_term

    self.prev_action = action.copy()
    return ..., r, done, ...
