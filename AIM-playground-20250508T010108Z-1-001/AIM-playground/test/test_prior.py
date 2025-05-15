import numpy as np
from latent_model import PriorNetwork

def test_prior_network_forward():
    prior = PriorNetwork(z_dim=32, action_dim=4)

    z_t = np.random.randn(4, 32).astype(np.float32)   # batch of 4
    a_t = np.random.randn(4, 4).astype(np.float32)    # 4D action space

    z_next, mean, log_std = prior(z_t, a_t)

    assert z_next.shape == (4, 32)
    assert mean.shape == (4, 32)
    assert log_std.shape == (4, 32)
    print("Prior network test passed.")

if __name__ == "__main__":
    test_prior_network_forward()
