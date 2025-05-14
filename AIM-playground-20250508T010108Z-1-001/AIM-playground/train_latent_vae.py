import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from latent_model import LatentModel

# === CONFIG ===
IMAGE_DIR = "/home/tariq/AIM_Task/AIM-playground-20250508T010108Z-1-001/AIM-playground/training_frames"
IMG_SIZE = (64, 64)
Z_DIM = 32
ACTION_DIM = 4
BATCH_SIZE = 32
EPOCHS = 45
SAVE_RECON_DIR = "vae_recon"

os.makedirs(SAVE_RECON_DIR, exist_ok=True)


# === 1. Dataset Loader ===
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def build_dataset(image_dir, batch_size):
    paths = tf.data.Dataset.list_files(os.path.join(image_dir, "*.png"), shuffle=True)
    ds = paths.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# === 2. Reconstruction Saver ===
def save_reconstructions(real, recon, epoch):
    for i in range(min(4, real.shape[0])):
        fig, axs = plt.subplots(1, 2, figsize=(4, 2))
        axs[0].imshow(real[i])
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(recon[i])
        axs[1].set_title("Reconstructed")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_RECON_DIR, f"epoch_{epoch:03d}_sample_{i}.png"))
        plt.close()


# === 3. Training Loop ===
def train_latent_vae():
    print("[INFO] Starting VAE training...")
    model = LatentModel(z_dim=Z_DIM, action_dim=ACTION_DIM)

    optimizer = tf.keras.optimizers.Adam(1e-3)
    dataset = build_dataset(IMAGE_DIR, BATCH_SIZE)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = tf.keras.metrics.Mean()
        for batch in dataset:
            batch_size = tf.shape(batch)[0]

            # Dummy z_t and action
            z_t = tf.random.normal((batch_size, Z_DIM))
            a_t = tf.random.normal((batch_size, ACTION_DIM))

            with tf.GradientTape() as tape:
                loss, components = model.compute_elbo(batch, a_t, z_t)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            avg_loss.update_state(loss)

        print(f"[EPOCH {epoch}] Loss: {avg_loss.result().numpy():.4f} | Recon: {components['recon_loss'].numpy():.4f} | KL: {components['kl_loss'].numpy():.4f}")

        # Save some reconstructions
        recon = components["reconstructed_x"].numpy()
        real = batch.numpy()
        save_reconstructions(real, recon, epoch)

    print(f"[INFO] Training complete. Reconstructions saved to {SAVE_RECON_DIR}/")


if __name__ == "__main__":
    train_latent_vae()
