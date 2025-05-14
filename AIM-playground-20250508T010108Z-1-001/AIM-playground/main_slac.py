import tensorflow as tf
import numpy as np

from slac_agent import SlacAgent
from actor_distribution_network import ActorDistributionNetwork
from critic_network import CriticNetwork
from compressor_network import Compressor
from model_distribution_network import ModelDistributionNetwork

from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.replay_buffers import TFUniformReplayBuffer
# from tf_agents.specs import tensor_spec
from tf_agents.specs import tensor_spec


# === ENV CONFIG ===
image_shape = (64, 64, 3)
action_dim = 2
sequence_length = 10
batch_size = 4

# === Load pretrained VAE weights (optional) ===
vae_checkpoint_dir = "./checkpoints_vae"

# === Network Instantiation ===
compressor = Compressor(base_depth=32)
model = ModelDistributionNetwork(
    observation_spec=image_shape,
    action_spec=(action_dim,),
    model_reward=True,
    model_discount=True
)
# actor = ActorDistributionNetwork(input_dim=128, output_dim=action_dim)  # placeholder dims
actor_input_spec = tf.TensorSpec(shape=(288,), dtype=tf.float32)  # latent or compressed image
# actor_output_spec = tf.TensorSpec(shape=(action_dim,), dtype=tf.float32)
actor_output_spec = tensor_spec.BoundedTensorSpec(
    shape=(action_dim,),
    dtype=tf.float32,
    minimum=-1.0,
    maximum=1.0
)

actor = ActorDistributionNetwork(
    input_tensor_spec=actor_input_spec,
    output_tensor_spec=actor_output_spec
)

# critic = CriticNetwork(input_dim=128 + action_dim)
critic_input_spec = (tf.TensorSpec(shape=(288,), dtype=tf.float32),  # state
                     tf.TensorSpec(shape=(action_dim,), dtype=tf.float32))  # action
critic = CriticNetwork(
    input_tensor_spec=critic_input_spec
)


# === Optimizers ===
actor_opt = tf.keras.optimizers.Adam(3e-4)
critic_opt = tf.keras.optimizers.Adam(3e-4)
alpha_opt = tf.keras.optimizers.Adam(3e-4)
model_opt = tf.keras.optimizers.Adam(3e-4)

# === Dummy specs ===
time_step_spec = ts.time_step_spec({
    'pixels': tf.TensorSpec(shape=image_shape, dtype=tf.float32),
    'state': tf.TensorSpec(shape=(4,), dtype=tf.float32)  # adjust if needed
})
action_spec = tf.TensorSpec(shape=(action_dim,), dtype=tf.float32)

# === SLAC Agent ===
agent = SlacAgent(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    critic_network=critic,
    actor_network=actor,
    model_network=model,
    compressor_network=compressor,
    actor_optimizer=actor_opt,
    critic_optimizer=critic_opt,
    alpha_optimizer=alpha_opt,
    model_optimizer=model_opt,
    sequence_length=sequence_length,
    actor_input='latent',
    critic_input='latent',
    trainable_model=False,
    train_step_counter=tf.Variable(0)
)

print("SLAC agent initialized successfully.")


# === Dummy Sanity Check ===
dummy_pixels = tf.random.uniform((batch_size, sequence_length + 1, *image_shape), dtype=tf.float32)
dummy_state = tf.random.uniform((batch_size, sequence_length + 1, 4), dtype=tf.float32)
dummy_actions = tf.random.uniform((batch_size, sequence_length, action_dim), dtype=tf.float32)
dummy_step_types = tf.constant([[ts.StepType.FIRST] + [ts.StepType.MID] * sequence_length] * batch_size)

(latent1, latent2), _ = model.sample_posterior(dummy_pixels, dummy_actions, dummy_step_types)
z = tf.concat([latent1, latent2], axis=-1)
print("Latent shape:", z.shape)  # Expect [B, T+1, D]

actor_input = z[:, -1]  # last latent
actor_step_type = dummy_step_types[:, -1]
action_dist, _ = actor(actor_input, actor_step_type)
sample_action = action_dist.sample()
print("Sampled action:", sample_action.shape)

critic_input = (actor_input, sample_action)
q_val, _ = critic(critic_input, actor_step_type)
print("Q-value:", q_val.numpy())