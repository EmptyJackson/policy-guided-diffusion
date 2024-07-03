import sys
import argparse


def parse_diffusion_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_nans", action="store_true")

    # Experiment
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Offline dataset name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_epochs", type=int, default=10000, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--eval_rate", type=int, default=50, help="Number of steps per evaluation"
    )

    # Dataset
    parser.add_argument(
        "--val_ratio", type=float, default=0.05, help="Validation ratio"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--trajectory_length",
        type=int,
        default=32,
        help="Trajectory length that the diffusion model will generate",
    )
    parser.add_argument(
        "--dataset_stride",
        type=int,
        default=8,
        help="Index stride for dataset sub-trajectory generation",
    )

    # Diffusion
    parser.add_argument(
        "--diffusion_method",
        type=str,
        default="edm",
        choices=["edm"],
        help="Diffusion method",
    )
    parser.add_argument(
        "--diffusion_timesteps",
        type=int,
        default=256,
        help="Number of timesteps for diffusion sampling",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help="Exponential moving average decay for model parameters",
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
        default=10,
        help="Number of steps between EMA updates",
    )

    # EDM
    parser.add_argument(
        "--edm_p_mean",
        type=float,
        default=-1.2,
        help="Mean of log-normal noise distribution",
    )
    parser.add_argument(
        "--edm_p_std",
        type=float,
        default=1.2,
        help="Standard deviation of log-normal noise distribution",
    )
    parser.add_argument(
        "--edm_sigma_data",
        type=float,
        default=1.0,
        help="Standard deviation of data distribution",
    )
    parser.add_argument(
        "--edm_sigma_min",
        type=float,
        default=0.002,
        help="Minimum noise level",
    )
    parser.add_argument(
        "--edm_sigma_max",
        type=float,
        default=80,
        help="Maximum noise level",
    )
    parser.add_argument(
        "--edm_rho",
        type=float,
        default=7.0,
        help="Sampling schedule",
    )
    parser.add_argument(
        "--edm_s_tmin",
        type=float,
        default=0.05,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_s_tmax",
        type=float,
        default=50.0,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_s_churn",
        type=float,
        default=80,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_s_noise",
        type=float,
        default=1.003,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_first_order",
        action="store_true",
        help="Use first-order Euler integration (disables second-order Heun)",
    )

    # U-Net
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=3,
        help="Number of blocks in the diffusion U-Net model",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=1024,
        help="Number of features in the diffusion U-Net model",
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")

    # Logging
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project")
    parser.add_argument("--wandb_team", type=str, default=None, help="WandB team")
    parser.add_argument("--wandb_group", type=str, default="debug", help="WandB group")

    args, rest_args = parser.parse_known_args(cmd_args)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    return args


def parse_agent_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_nans", action="store_true")

    # Experiment
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Offline dataset name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1_000_000,
        help="Number of epochs or agent train steps",
    )
    parser.add_argument(
        "--eval_rate",
        type=int,
        default=1,
        help="Number of train steps between evaluations",
    )
    parser.add_argument(
        "--num_env_workers",
        type=int,
        default=16,
        help="Number of environment workers for evaluation",
    )
    parser.add_argument("--batch_size", type=int, default=256)

    # Synthetic experience
    parser.add_argument("--synthetic_experience", action="store_true")
    parser.add_argument(
        "--num_synth_workers",
        type=int,
        default=32,
        help="Number of parallel workers for synthetic rollout",
    )
    parser.add_argument(
        "--num_synth_rollouts",
        type=int,
        default=256,
        help="Number of synthetic rollouts per worker",
    )
    parser.add_argument(
        "--synth_dataset_lifetime",
        type=int,
        default=10000,
        help="Number of steps before synthetic dataset is resampled",
    )
    parser.add_argument(
        "--synth_batch_size",
        type=int,
        default=240,
        help="Number of synthetic samples to use per-batch",
    )
    parser.add_argument(
        "--synth_batch_lifetime",
        type=int,
        default=1,
        help="Number of epochs before a synthetic batch is resampled",
    )
    parser.add_argument("--diffusion_timesteps", type=int, default=None)
    parser.add_argument("--denoiser_checkpoint", type=str, default=None)

    # Policy guidance
    parser.add_argument("--policy_guidance_coeff", type=float, default=0.0)
    parser.add_argument("--policy_guidance_cosine_coeff", type=float, default=0.3)
    parser.add_argument(
        "--normalize_action_guidance",
        action="store_true",
        help="Normalize action guidance",
    )
    parser.add_argument(
        "--denoised_guidance",
        action="store_true",
        help="Apply guidance to denoised trajectory",
    )

    # Agent
    parser.add_argument(
        "--agent", type=str, default="iql", choices=["iql", "td3_bc"], help="Agent type"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function for actor critic",
    )
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        default=128,
        help="Number of rollout steps per agent update",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--value_loss_coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--polyak_step_size",
        type=float,
        default=0.005,
        help="Target update step size",
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr_schedule", type=str, default="constant")

    # TD3+BC
    parser.add_argument(
        "--policy_noise", type=float, default=0.2, help="Policy noise parameter"
    )
    parser.add_argument(
        "--noise_clip", type=float, default=0.5, help="Noise clip parameter"
    )
    parser.add_argument("--a_max", type=float, default=1.0, help="Maximum action value")
    parser.add_argument(
        "--num_critic_updates_per_step",
        type=int,
        default=2,
        help="Number of critic updates per step",
    )
    parser.add_argument(
        "--td3_alpha", type=float, default=2.5, help="TD3 alpha parameter"
    )
    parser.add_argument(
        "--normalize_obs", action="store_true", help="Normalize observations"
    )

    # IQL
    parser.add_argument(
        "--iql_tau", type=float, default=0.7, help="Asymmetric L2 loss parameter"
    )
    parser.add_argument(
        "--iql_beta", type=float, default=3.0, help="Advantage scaling parameter"
    )

    # Logging
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--wandb_project", default=None, type=str, help="WandB project")
    parser.add_argument("--wandb_team", default=None, type=str, help="WandB team")
    parser.add_argument("--wandb_group", type=str, default="debug", help="Wandb group")

    args, rest_args = parser.parse_known_args(cmd_args)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    assert (
        not args.synthetic_experience
        or args.num_train_steps % args.synth_dataset_lifetime == 0
    ), "Number of train steps must be a multiple of the synthetic dataset lifetime"
    args.env_name = args.dataset_name
    return args
