import numpy as np
from latent_model import SLACEncoder

def test_encoder_forward():
    encoder = SLACEncoder(z_dim=32, input_channels=9)

    # Create dummy input: batch of 4, 3 stacked RGB images
    dummy_input = np.random.rand(4, 64, 64, 9).astype(np.float32)

    z, mean, log_std = encoder(dummy_input)

    assert z.shape == (4, 32), f"Expected z shape (4, 32), got {z.shape}"
    assert mean.shape == (4, 32), f"Expected mean shape (4, 32), got {mean.shape}"
    assert log_std.shape == (4, 32), f"Expected log_std shape (4, 32), got {log_std.shape}"
    print("Encoder test passed.")

if __name__ == "__main__":
    test_encoder_forward()
