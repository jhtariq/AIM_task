from train_eval import train_eval
import functools
import tensorflow as tf
import model_distribution_network
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


if __name__ == '__main__':
    train_eval(
        root_dir='./logs',
        experiment_name='arm_slac_test',
        train_eval_dir=None,
        universe=None,
        env_name=None,
        domain_name=None,
        task_name=None,
        action_repeat=2,
        num_iterations=50,
        actor_fc_layers=(256, 256),
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=(256, 256),
        model_network_ctor=model_distribution_network.ModelDistributionNetwork,
        critic_input='latent',
        actor_input='latent',
        compressor_descriptor='preprocessor_32_3',
        initial_collect_steps=100,
        collect_steps_per_iteration=1,
        replay_buffer_capacity=500,
        target_update_tau=0.005,
        target_update_period=1,
        train_steps_per_iteration=1,
        model_train_steps_per_iteration=1,
        initial_model_train_steps=45,
        batch_size=8,
        model_batch_size=4,
        sequence_length=3,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        model_learning_rate=1e-4,
        td_errors_loss_fn=functools.partial(
            tf.compat.v1.losses.mean_squared_error, weights=0.5),
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        num_eval_episodes=2,
        eval_interval=10,
        num_images_per_summary=1,
        train_checkpoint_interval=10,
        policy_checkpoint_interval=10,
        rb_checkpoint_interval=10,
        log_interval=1,
        summary_interval=1,
        summaries_flush_secs=5,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        gpu_allow_growth=True,
        gpu_memory_limit=None
    )
