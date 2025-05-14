import numpy as np
from latent_model import LatentModel

def test_latent_model_elbo():
    model = LatentModel(z_dim=32, action_dim=4)

    x_tp1 = np.random.rand(4, 64, 64, 3).astype(np.float32)
    a_t = np.random.rand(4, 4).astype(np.float32)
    z_t = np.random.rand(4, 32).astype(np.float32)

    loss, components = model.compute_elbo(x_tp1, a_t, z_t)

    print(f"Loss: {loss.numpy():.4f}")
    print("Recon loss:", components["recon_loss"].numpy())
    print("KL loss:", components["kl_loss"].numpy())
    print("Latent model ELBO works.")

if __name__ == "__main__":
    test_latent_model_elbo()
