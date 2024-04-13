import optax
import jax.numpy as jnp

from flax.training.train_state import TrainState
from functools import partial

import diffusion.edm as edm
from models.diffusion import UNet
from util import *


def create_denoiser_train_state(rng, obs_dim, action_dim, args, dataset_len):
    # --- Create U-Net model ---
    denoiser = UNet(args.num_features, args.num_blocks)
    placeholder_batch, placeholder_seq = 2, 64
    denoiser_params = denoiser.init(
        rng,
        jnp.ones((placeholder_batch, placeholder_seq, obs_dim + action_dim + 2)),
        jnp.ones((1,)),
    )

    # --- Create cosine decay schedule ---
    num_steps_per_epoch = dataset_len // args.batch_size
    total_steps = num_steps_per_epoch * args.num_epochs
    warmup_steps = total_steps // 10
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
    )

    # --- Return train state ---
    return TrainState.create(
        apply_fn=denoiser.apply,
        params=denoiser_params,
        tx=optax.adam(learning_rate=lr_schedule),
    )


def get_denoiser_hypers(args):
    if args.diffusion_method == "edm":
        return edm.DenoiserHyperparams(
            p_mean=args.edm_p_mean,
            p_std=args.edm_p_std,
            sigma_data=args.edm_sigma_data,
            sigma_min=args.edm_sigma_min,
            sigma_max=args.edm_sigma_max,
            rho=args.edm_rho,
            edm_first_order=args.edm_first_order,
            diffusion_timesteps=args.diffusion_timesteps,
            s_tmin=args.edm_s_tmin,
            s_tmax=args.edm_s_tmax,
            s_churn=args.edm_s_churn,
            s_noise=args.edm_s_noise,
        )
    raise ValueError(f"Unknown diffusion method {args.diffusion_method}.")


def make_train_step(args):
    hypers = get_denoiser_hypers(args)
    if args.diffusion_method == "edm":
        return partial(edm.train_step, denoiser_hyperparams=hypers)
    raise ValueError(f"Unknown diffusion method {args.diffusion_method}.")


def make_sample_fn(
    args,
    normalize_action_guidance,
    denoised_guidance,
    det_guidance,
):
    hypers = get_denoiser_hypers(args)
    if args.diffusion_method == "edm":
        return partial(
            edm.sample_trajectory,
            denoiser_hyperparams=hypers,
            normalize_action_guidance=normalize_action_guidance,
            denoised_guidance=denoised_guidance,
            det_guidance=det_guidance,
        )
    raise ValueError(f"Unknown diffusion method {args.diffusion_method}.")
