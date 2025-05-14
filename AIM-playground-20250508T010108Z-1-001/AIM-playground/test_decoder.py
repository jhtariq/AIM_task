import numpy as np
from latent_model import Decoder
import tensorflow as tf

def test_decoder_forward():
    decoder = Decoder(z_dim=32, output_shape=(64, 64, 3))

    z = np.random.randn(4, 32).astype(np.float32)
    recon = decoder(z)

    assert recon.shape == (4, 64, 64, 3), f"Got shape {recon.shape}"
    assert tf.reduce_max(recon).numpy() <= 1.0
    assert tf.reduce_min(recon).numpy() >= 0.0
    print("Decoder test passed.")

if __name__ == "__main__":
    test_decoder_forward()
