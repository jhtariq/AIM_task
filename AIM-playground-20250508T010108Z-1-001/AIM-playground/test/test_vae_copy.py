import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_agents.trajectories import time_step as ts
from model_distribution_network import ModelDistributionNetwork

# Load images
def load_image_sequence(folder, sequence_length=10):
    image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])
    image_files = image_files[:sequence_length]
    images = [tf.image.decode_image(tf.io.read_file(f), channels=3) for f in image_files]
    images = [tf.image.resize(img, [64, 64]) for img in images]
    images = tf.stack(images)
    images = tf.cast(images, tf.float32) / 255.0
    return tf.expand_dims(images, axis=0)  # [1, T, H, W, C]

# Setup
folder = "./training_frames"
batch_size = 1
sequence_length = 10
action_dim = 2

images = load_image_sequence(folder, sequence_length)
actions = tf.zeros((batch_size, sequence_length, action_dim), dtype=tf.float32)
step_types = [ts.StepType.FIRST] + [ts.StepType.MID] * (sequence_length - 2) + [ts.StepType.LAST]
step_types = tf.convert_to_tensor([step_types], dtype=tf.int32)

# Model
vae = ModelDistributionNetwork(
    observation_spec=(64, 64, 3),
    action_spec=(2,),
    model_reward=False,
    model_discount=False
)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Setup checkpointing
checkpoint_dir = './checkpoints_vae'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=vae, optimizer=optimizer)

# Training loop using built-in ELBO loss
epochs = 100
# For tracking
elbo_list = []
kl_list = []
recon_error_list = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss, outputs = vae.compute_loss(images, actions, step_types)
    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))

    elbo = outputs['elbo'].numpy()
    kl = outputs['kl_divergence'].numpy()
    recon = outputs['reconstruction_error'].numpy()

    elbo_list.append(elbo)
    kl_list.append(kl)
    recon_error_list.append(recon)

    print(f"Epoch {epoch+1}, ELBO: {elbo:.2f}, KL: {kl:.2f}, Recon: {recon:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(elbo_list, label="ELBO")
plt.plot(kl_list, label="KL Divergence")
plt.plot(recon_error_list, label="Reconstruction Error")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("ELBO, KL Divergence, and Reconstruction Error over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final forward pass for visualization
_, outputs = vae.compute_loss(images, actions, step_types)
# print("Available output keys:", outputs.keys())
# posterior_latents = outputs["posterior_latents"][0].numpy()

# sequence_len, latent_dim = posterior_latents.shape

# plt.figure(figsize=(16, 6))
# for i in range(min(latent_dim, 10)):
#     plt.plot(range(sequence_len), posterior_latents[:, i], label=f"z_{i}")
# plt.xlabel("Timestep")
# plt.ylabel("Latent value")
# plt.title("Posterior Latent Vectors over Time")
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=1)
# plt.tight_layout()
# plt.show()

# posterior_latents = outputs["posterior_latents"][0].numpy()  # shape [T, latent_dim]
# sequence_len, latent_dim = posterior_latents.shape

# plt.figure(figsize=(16, 6))
# for i in range(latent_dim):
#     plt.plot(range(sequence_len), posterior_latents[:, i], label=f"z_{i}")
# plt.xlabel("Timestep")
# plt.ylabel("Latent value")
# plt.title("Posterior Latent Vectors over Time")
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=1)
# plt.tight_layout()
# plt.show()

# Visualize
# original = images[0].numpy()
# recon = outputs["posterior_images"][0].numpy()

# for i in range(sequence_length):
#     plt.subplot(2, sequence_length, i + 1)
#     plt.imshow(original[i])
#     plt.axis("off")
#     plt.subplot(2, sequence_length, sequence_length + i + 1)
#     plt.imshow(recon[i])
#     plt.axis("off")

# plt.suptitle("Original (top) vs Reconstruction (bottom)")
# plt.show()
