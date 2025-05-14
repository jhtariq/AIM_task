from __future__ import absolute_import, division, print_function

import os
import time
import tensorflow as tf
import gin
import re
import tensorflow as tf
import numpy as np
from absl import app, flags, logging
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
# from tf_agents.metrics import py_metric, py_metrics, tf_metrics, tf_py_metric
from tf_agents.metrics import tf_metrics

from tf_agents.policies import greedy_policy, py_tf_policy, random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils

from arm_env_wrapper import ArmPyEnvWrapper as ArmEnvWrapper

import actor_distribution_network
import critic_network
import compressor_network
import model_distribution_network
import slac_agent
import gif_utils  # You mentioned keeping TensorBoard summaries
import functools
from tf_agents.eval import metric_utils

# tf.compat.v1.disable_eager_execution()
nest = tf.nest
summary_writer = tf.summary.create_file_writer('./logs')
summary_writer.set_as_default()

# Define flags
flags.DEFINE_string('root_dir', None, 'Root directory for logs/checkpoints.')
flags.DEFINE_string('experiment_name', None, 'Experiment name.')
flags.DEFINE_string('train_eval_dir', None, 'Override for output directory.')
flags.DEFINE_multi_string('gin_file', None, 'Path to Gin config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin parameters to override.')

FLAGS = flags.FLAGS
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

def get_train_eval_dir(root_dir, experiment_name):
    """Returns the directory for saving train/eval logs."""
    root_dir = os.path.expanduser(root_dir)
    return os.path.join(root_dir, experiment_name)


def pad_and_concatenate_videos(videos):
    """Pads and concatenates episodes (frames) for summary display."""
    max_episode_length = max(len(video) for video in videos)
    for video in videos:
        if len(video) < max_episode_length:
            video.extend([np.zeros_like(video[-1])] * (max_episode_length - len(video)))
    return [np.concatenate(frames, axis=1) for frames in zip(*videos)]


def compute_summaries(metrics, tf_env, policy, num_episodes):
    """
    Evaluates the policy on the tf_env and logs the metric summaries.

    Args:
        metrics: List of TF metrics (e.g., AverageReturnMetric).
        tf_env: A TFPyEnvironment.
        policy: A TFPolicy (e.g., tf_agent.policy or GreedyPolicy(tf_agent.policy)).
        num_episodes: Number of episodes to run evaluation for.
    """
    # Reset all metrics
    for metric in metrics:
        metric.reset()

    time_step = tf_env.reset()
    policy_state = policy.get_initial_state(batch_size=tf_env.batch_size)

    episode_count = 0

    while episode_count < num_episodes:
        action_step = policy.action(time_step, policy_state)
        next_time_step = tf_env.step(action_step.action)
        # print("STEP obs max:", np.max(obs), "min:", np.min(obs), "mean:", np.mean(obs))
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        for observer in metrics:
            observer(traj)

        # Count completed episodes
        episode_count += tf.reduce_sum(tf.cast(traj.is_last(), tf.int32)).numpy()

        # Advance the environment state
        time_step = next_time_step
        policy_state = action_step.state

    # Write summaries for the evaluation metrics
    for metric in metrics:
        metric.tf_summaries(train_step=tf.compat.v1.train.get_global_step(), step_metrics=[])



def get_control_timestep(py_env):
    """Returns timestep used by the environment."""
    try:
        return py_env.dt  # e.g., gym-style envs
    except AttributeError:
        # return py_env.control_timestep()  # dm_control-style
        return 0.1




@gin.configurable
def train_eval(
    root_dir,
    experiment_name,
    train_eval_dir=None,
    universe='gym',
    env_name='HalfCheetah-v2',
    domain_name='cheetah',
    task_name='run',
    action_repeat=1,
    num_iterations=int(40),
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    model_network_ctor=model_distribution_network.ModelDistributionNetwork,
    critic_input='state',
    actor_input='state',
    compressor_descriptor='preprocessor_32_3',
    # Params for collect
    initial_collect_steps=10000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=int(1e5),
    # increase if necessary since buffers with images are huge
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    train_steps_per_iteration=1,
    model_train_steps_per_iteration=1,
    initial_model_train_steps=100000,
    batch_size=256,
    model_batch_size=32,
    sequence_length=4,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    model_learning_rate=1e-4,
    td_errors_loss_fn=functools.partial(
        tf.compat.v1.losses.mean_squared_error, weights=0.5),
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=5,
    # Params for summaries and logging
    num_images_per_summary=1,
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=0, # enable if necessary since buffers with images are huge
    log_interval=10,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    gpu_allow_growth=False,
    gpu_memory_limit=None):
    """A simple train and eval setup for SLAC."""


    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if gpu_allow_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            if gpu_memory_limit:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)]
                    )
        except RuntimeError as e:
            print("[WARNING] GPU memory config must be set before GPUs are initialized. Skipping. Error:", e)


    # Setup directories
    if train_eval_dir is None:
        train_eval_dir = get_train_eval_dir(root_dir, experiment_name)

    train_dir = os.path.join(train_eval_dir, 'train')
    eval_dir = os.path.join(train_eval_dir, 'eval')

    # TensorBoard writers
    train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(name='AverageReturnEvalPolicy', buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(name='AverageEpisodeLengthEvalPolicy', buffer_size=num_eval_episodes),
    ]
    eval_greedy_metrics = [
        tf_metrics.AverageReturnMetric(name='AverageReturnEvalGreedyPolicy', buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(name='AverageEpisodeLengthEvalGreedyPolicy', buffer_size=num_eval_episodes),
    ]
    eval_summary_flush_op = eval_summary_writer.flush()

    global_step = tf.compat.v1.train.get_or_create_global_step()

    with tf.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        # === Environment Setup ===
        trainable_model = model_train_steps_per_iteration != 0
        state_only = (
            actor_input == 'state' and
            critic_input == 'state' and
            not trainable_model and
            initial_model_train_steps == 0
        )
        observations_whitelist = ['state'] if state_only else None

        py_env = ArmEnvWrapper(image_size=(64, 64))
        # print("[DEBUG - PyEnv] obs shape:", obs_py.shape, "max:", np.max(obs_py), "mean:", np.mean(obs_py))

        eval_py_env = ArmEnvWrapper(image_size=(64, 64))
        tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=True)
        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env, isolation=True)
        # print("[DEBUG - TFPyEnv] obs shape:", obs_tf.shape, "max:", tf.reduce_max(obs_tf), "mean:", tf.reduce_mean(obs_tf))

        original_control_timestep = get_control_timestep(eval_py_env)
        control_timestep = original_control_timestep * float(action_repeat)
        fps = int(np.round(1.0 / control_timestep))
        render_fps = int(np.round(1.0 / original_control_timestep))

        time_step_spec = tf_env.time_step_spec()
        observation_spec = time_step_spec.observation
        action_spec = tf_env.action_spec()

        if model_train_steps_per_iteration not in (0, train_steps_per_iteration):
            raise NotImplementedError("Mismatch in model training and policy training steps.")

        model_net = model_network_ctor(observation_spec, action_spec)

        # === Compressor Selection ===
        if compressor_descriptor == 'model':
            compressor_net = model_net.compressor
        elif re.match(r'preprocessor_(\d+)_(\d+)', compressor_descriptor):
            filters, n_layers = map(int, re.match(r'preprocessor_(\d+)_(\d+)', compressor_descriptor).groups())
            compressor_net = compressor_network.Preprocessor(filters, n_layers=n_layers)
        elif re.match(r'compressor_(\d+)', compressor_descriptor):
            filters = int(re.match(r'compressor_(\d+)', compressor_descriptor).group(1))
            compressor_net = compressor_network.Compressor(filters)
        elif re.match(r'softlearning_(\d+)_(\d+)', compressor_descriptor):
            filters, n_layers = map(int, re.match(r'softlearning_(\d+)_(\d+)', compressor_descriptor).groups())
            compressor_net = compressor_network.SoftlearningPreprocessor(filters, n_layers=n_layers)
        elif compressor_descriptor == 'd4pg':
            compressor_net = compressor_network.D4pgPreprocessor()
        else:
            raise NotImplementedError(f"Unsupported compressor: {compressor_descriptor}")

        # === Actor Input Spec Construction ===
        actor_state_size = 0
        for _actor_input in actor_input.split('__'):
            if _actor_input == 'state':
                state_size = observation_spec['state'].shape[0]
                actor_state_size += state_size
            elif _actor_input == 'latent':
                actor_state_size += model_net.state_size
            elif _actor_input == 'feature':
                actor_state_size += compressor_net.feature_size
            elif _actor_input in ('sequence_feature', 'sequence_action_feature'):
                actor_state_size += compressor_net.feature_size * sequence_length
                if _actor_input == 'sequence_action_feature':
                    actor_state_size += action_spec.shape[0] * (sequence_length - 1)
            else:
                raise NotImplementedError(f"Unknown actor input: {_actor_input}")

        actor_input_spec = tensor_spec.TensorSpec((actor_state_size,), dtype=tf.float32)

        # === Critic Input Spec Construction ===
        critic_state_size = 0
        for _critic_input in critic_input.split('__'):
            if _critic_input == 'state':
                state_size = observation_spec['state'].shape[0]
                critic_state_size += state_size
            elif _critic_input == 'latent':
                critic_state_size += model_net.state_size
            elif _critic_input == 'feature':
                critic_state_size += compressor_net.feature_size
            elif _critic_input in ('sequence_feature', 'sequence_action_feature'):
                critic_state_size += compressor_net.feature_size * sequence_length
                if _critic_input == 'sequence_action_feature':
                    critic_state_size += action_spec.shape[0] * (sequence_length - 1)
            else:
                raise NotImplementedError(f"Unknown critic input: {_critic_input}")

        critic_input_spec = tensor_spec.TensorSpec((critic_state_size,), dtype=tf.float32)

        # === Actor & Critic Networks ===
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            actor_input_spec,
            action_spec,
            fc_layer_params=actor_fc_layers
        )
        critic_net = critic_network.CriticNetwork(
            (critic_input_spec, action_spec),
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers
        )

    tf_agent = slac_agent.SlacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        model_network=model_net,
        compressor_network=compressor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
        model_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=model_learning_rate),
        sequence_length=sequence_length,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        trainable_model=trainable_model,
        critic_input=critic_input,
        actor_input=actor_input,
        model_batch_size=model_batch_size,
        control_timestep=control_timestep,
        num_images_per_summary=num_images_per_summary,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step
    )



    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity
    )
    replay_observer = [replay_buffer.add_batch]

    # eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)
    # eval_greedy_py_policy = py_tf_policy.PyTFPolicy(greedy_policy.GreedyPolicy(tf_agent.policy))
    eval_py_policy = tf_agent.policy
    eval_greedy_py_policy = greedy_policy.GreedyPolicy(tf_agent.policy)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(buffer_size=1),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=1)
    ]

    collect_policy = tf_agent.collect_policy
    initial_collect_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)
    initial_policy_state = initial_collect_policy.get_initial_state(tf_env.batch_size)

    def log_observer(traj):
        pixels = traj.observation['pixels']
        # print(f"{BLUE}[BUFFER OBS] shape:", pixels.shape, "max:", tf.reduce_max(pixels), "mean:", tf.reduce_mean(pixels))

    
    initial_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=replay_observer + train_metrics,
    num_steps=initial_collect_steps  # at least sequence_length + 1
)
# Collects initial data to fill the replay buffer
    # initial_driver = dynamic_step_driver.DynamicStepDriver(
    #     tf_env,
    #     initial_collect_policy,
    #     observers=[replay_buffer.add_batch, log_observer] + train_metrics,
    #     num_steps=initial_collect_steps
    # )

    final_time_step, final_policy_state = initial_driver.run(
    policy_state=initial_policy_state
)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    collect_policy,
    observers=replay_observer + train_metrics,
    num_steps=collect_steps_per_iteration
)
#     collect_driver = dynamic_step_driver.DynamicStepDriver(
#     tf_env,
#     collect_policy,
#     observers=[replay_buffer.add_batch, log_observer] + train_metrics,
#     num_steps=collect_steps_per_iteration
# )
    # 4. Now it is safe to sample from the dataset
    def _filter_invalid_transition(trajectories, unused_arg1):
        return ~trajectories.is_boundary()[-2]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=sequence_length + 1
    ).unbatch().filter(
        _filter_invalid_transition
    ).batch(batch_size, drop_remainder=True).prefetch(3)
    #comment to say that sampling is done in the dataset
    print(f"{YELLOW}[DEBUG] Dataset created, sampling is done in the dataset")
    dataset_iterator = iter(dataset)
    trajectories, _ = next(dataset_iterator)
    print(f"{RED}[TRAINING BATCH] obs shape:", trajectories.observation['pixels'].shape)
    print("max:", tf.reduce_max(trajectories.observation['pixels']), "mean:", tf.reduce_mean(trajectories.observation['pixels']))
    train_op = tf_agent.train(trajectories)

    summary_ops = [
        metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])
        for metric in train_metrics
    ]

    if initial_model_train_steps:
        with tf.name_scope('initial'):
            model_train_op = tf_agent.train_model(trajectories)
            model_summary_ops = []
            all_ops = tf.compat.v1.summary.all_v2_summary_ops()
            if all_ops is not None:
                for op in all_ops:
                    if op not in summary_ops:
                        model_summary_ops.append(op)

    with eval_summary_writer.as_default(), \
         tf.compat.v2.summary.record_if(True):
      for eval_metric in eval_metrics + eval_greedy_metrics:
        eval_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

    train_config_saver = gin.tf.GinConfigSaverHook(
        train_dir, summarize_config=False)
    eval_config_saver = gin.tf.GinConfigSaverHook(
        eval_dir, summarize_config=False)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        max_to_keep=2)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=tf_agent.policy,
        global_step=global_step,
        max_to_keep=2)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)
        
    # === Restore Checkpoints (no session needed in TF2) ===
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()
    # Manually reset if stale
    if global_step.numpy() >= initial_model_train_steps + num_iterations:
        print(f"[WARN] Resetting global_step from {global_step.numpy()} to 0")
        global_step.assign(0)

    # === Initial Data Collection ===
    if global_step.numpy() == 0:
        logging.info('Global step 0: Running initial collect.')
        final_time_step, final_policy_state = initial_driver.run()
        rb_checkpointer.save(global_step=global_step.numpy())
        logging.info('Initial data collection complete.')
    else:
        logging.info(f'Global step {global_step.numpy()}: Skipping initial collect.')

    # === Training Loop ===
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0

    for iteration in range(global_step.numpy(), initial_model_train_steps + num_iterations):
        start_time = time.time()
        #print current iteration
        print(f"{MAGENTA}[DEBUG] Iteration {iteration} / {initial_model_train_steps + num_iterations}")

        if iteration < initial_model_train_steps:
            total_loss = tf_agent.train_model(next(dataset_iterator)[0])
        else:
            final_time_step, policy_state = collect_driver.run(
                time_step=final_time_step,
                policy_state=policy_state
            )
            for _ in range(train_steps_per_iteration):
                total_loss = tf_agent.train(next(dataset_iterator)[0])

        # Increment global step
        global_step.assign_add(1)
        time_acc += time.time() - start_time

        if log_interval and iteration % log_interval == 0:
            steps_per_sec = (iteration - timed_at_step) / time_acc
            timed_at_step = iteration
            time_acc = 0
            logging.info(f'step = {iteration}, loss = {total_loss.loss}, {steps_per_sec:.2f} steps/sec')

        if train_checkpoint_interval and iteration % train_checkpoint_interval == 0:
            train_checkpointer.save(global_step=global_step)

        if policy_checkpoint_interval and iteration % policy_checkpoint_interval == 0:
            policy_checkpointer.save(global_step=global_step)

        if rb_checkpoint_interval and iteration % rb_checkpoint_interval == 0:
            rb_checkpointer.save(global_step=global_step)

        if eval_interval and iteration % eval_interval == 0:
            logging.info(f"[Eval] Running evaluation at step {iteration}")
            for _eval_metrics, _eval_policy in [
                    (eval_metrics, eval_py_policy),
                    (eval_greedy_metrics, eval_greedy_py_policy)
                ]:
                with eval_summary_writer.as_default():
                    compute_summaries(_eval_metrics, eval_tf_env, _eval_policy, num_eval_episodes)
                    eval_summary_writer.flush()






    






