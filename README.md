# AIM_task
DDPG Improvization Task
# SLAC (Stochastic Latent Actor-Critic) for Arm Environment

This project implements the SLAC reinforcement learning algorithm on a custom 2-link robotic arm environment using image-based observations.

## 🚀 How to Run

To train and evaluate the SLAC agent:

1. Open `run_slac.py` and set your desired training parameters (e.g., environment config, model type, latent dimension, etc.).

2. Run the script:

```bash
python run_slac.py



Notes:

The current implementation does not use a pre-trained VAE for running SLAC. There is a code file under the tests folder, which can be used to train the VAE. The folder name is test_vae_copy. You would need to set the number of epochs and then run the comman python test_vae_copy.py. The checkpoints can be saved and imported before running SLAC. 
