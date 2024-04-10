import os
import wandb

RESOURCE_DIR = "diffusion-rl/res"
CHECKPOINT_DIR = os.path.join(RESOURCE_DIR, "checkpoints")
TRAJECTORY_DIR = os.path.join(RESOURCE_DIR, "trajectories")
MAX_LOG_STEPS = 5000


def log(log_dict):
    wandb.log(log_dict)
