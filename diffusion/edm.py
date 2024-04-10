import jax
import jax.numpy as jnp
from flax import struct
from distrax import Normal

from util import *


@struct.dataclass
class DenoiserHyperparams:
    p_mean: float = -1.2  # Mean of log-normal noise distribution
    p_std: float = 1.2  # Standard deviation of log-normal noise distribution
    sigma_data: float = 1.0  # Standard deviation of data distribution
    sigma_min: float = 0.002  # Minimum noise level
    sigma_max: int = 80  # Maximum noise level
    rho: float = 7.0  # Sampling schedule
    edm_first_order: bool = False  # Disable second order Heun integration
    diffusion_timesteps: int = 200  # Number of diffusion timesteps for sampling
    # Stochastic sampling coefficients
    s_tmin: float = 0.05
    s_tmax: float = 50.0
    s_churn: float = 80.0
    s_noise: float = 1.003


# Derived preconditioning params - EDM Table 1
def c_skip(sigma, sigma_data):
    return (sigma_data**2) / (sigma**2 + sigma_data**2)


def c_out(sigma, sigma_data):
    return sigma * sigma_data * ((sigma_data**2 + sigma**2) ** -0.5)


def c_in(sigma, sigma_data):
    return (sigma**2 + sigma_data**2) ** -0.5


def c_noise(sigma):
    return jnp.log(sigma) * 0.25


def train_step(
    rng,
    batch,
    denoiser_state,
    denoiser_hyperparams,
):
    """
    Params:
        data: (batch, seq_len, obs_dim + action_dim + 2)
        ts: (batch,)
    """

    def loss_weight(sigma):
        return (sigma**2 + denoiser_hyperparams.sigma_data**2) * (
            sigma * denoiser_hyperparams.sigma_data
        ) ** -2

    def seq_loss(denoiser_params, rng, seq):
        # Implements EDM from https://openreview.net/pdf?id=k7FuTOWMOc7
        rng, _rng = jax.random.split(rng)
        sigma = jnp.exp(
            (
                denoiser_hyperparams.p_mean
                + denoiser_hyperparams.p_std * jax.random.normal(_rng)
            )
        )

        rng, _rng = jax.random.split(rng)
        noise = jax.random.normal(_rng, shape=seq.shape)
        noised_seq = seq + sigma * noise  # alphas are 1. in the paper
        noise_pred = denoiser_state.apply_fn(
            denoiser_params,
            c_in(sigma, denoiser_hyperparams.sigma_data) * noised_seq,
            c_noise(sigma),
        )
        denoised_pred = (
            c_skip(sigma, denoiser_hyperparams.sigma_data) * noised_seq
            + c_out(sigma, denoiser_hyperparams.sigma_data) * noise_pred
        )
        return jnp.square(denoised_pred - seq) * loss_weight(sigma)

    def batch_loss(denoiser_params):
        _rng = jax.random.split(rng, batch.shape[0])
        return jnp.mean(
            jax.vmap(seq_loss, in_axes=(None, 0, 0))(denoiser_params, _rng, batch)
        )

    loss_val, grad = jax.value_and_grad(batch_loss)(denoiser_state.params)
    denoiser_state = denoiser_state.apply_gradients(grads=grad)
    return denoiser_state, loss_val


def sample_trajectory(
    rng,
    denoiser_state,
    seq_len,
    obs_dim,
    action_dim,
    denoiser_norm_stats,
    denoiser_hyperparams,
    policy_guidance_coeff=0.0,
    policy_guidance_delay_steps=0,
    policy_guidance_cosine_coeff=0.0,
    normalize_action_guidance=True,
    normalize_reward=False,
    denoised_guidance=False,
    det_guidance=False,
    agent_apply_fn=None,
    agent_params=None,
):
    # --- Compute noise schedule ---
    def _get_noise_schedule(num_diffusion_timesteps):
        inv_rho = 1 / denoiser_hyperparams.rho
        sigmas = (
            denoiser_hyperparams.sigma_max**inv_rho
            + (jnp.arange(num_diffusion_timesteps + 1) / (num_diffusion_timesteps - 1))
            * (
                denoiser_hyperparams.sigma_min**inv_rho
                - denoiser_hyperparams.sigma_max**inv_rho
            )
        ) ** denoiser_hyperparams.rho
        return sigmas.at[-1].set(0.0)  # last step has sigma value of 0.

    sigmas = _get_noise_schedule(denoiser_hyperparams.diffusion_timesteps)
    gammas = jnp.where(
        (sigmas >= denoiser_hyperparams.s_tmin)
        & (sigmas <= denoiser_hyperparams.s_tmax),
        jnp.minimum(
            denoiser_hyperparams.s_churn / denoiser_hyperparams.diffusion_timesteps,
            jnp.sqrt(2) - 1,
        ),
        0.0,
    )

    # --- Sample random noise trajecory ---
    rng, _rng = jax.random.split(rng)
    # Add 2 dimensions for reward and done
    init_noise = jax.random.normal(_rng, (seq_len, obs_dim + action_dim + 2))
    init_noise *= sigmas[0]

    # --- Construct guidance function ---
    do_apply_guidance = (
        agent_apply_fn is not None
        and agent_params is not None
        and policy_guidance_coeff != 0.0
    )

    def _compute_action_guidance(traj):
        # --- Unnormalize observation ---
        obs = traj[:, :obs_dim]
        obs = unnormalise_traj(obs, denoiser_norm_stats["obs"])

        # --- Compute guidance from policy ---
        pi = agent_apply_fn(agent_params, obs)
        if det_guidance:
            # Apply guidance to unit Gaussian around deterministic action
            agent_action = pi.sample(seed=jax.random.PRNGKey(0))
            pi = Normal(agent_action, 1.0)

        def _transformed_action_log_prob(action):
            action = unnormalise_traj(action, denoiser_norm_stats["action"])
            action = jnp.tanh(action)
            return pi.log_prob(action).sum()

        action = traj[:, obs_dim : obs_dim + action_dim]
        action_guidance = jax.grad(_transformed_action_log_prob)(action)

        # --- Normalize and return guidance ---
        if normalize_action_guidance:
            action_guidance = action_guidance / jnp.linalg.norm(action_guidance) + 1e-8
        return action_guidance

    def denoise_step(runner_state, step_coeffs):
        rng, noised_traj, step_idx = runner_state
        sigma, next_sigma, gamma = step_coeffs

        if do_apply_guidance:
            # --- Compute guidance coefficient ---
            n_steps = denoiser_hyperparams.diffusion_timesteps
            lambd = 1.0 - (step_idx / n_steps)
            cosine_adjustment = jnp.sin(jnp.pi * ((step_idx + 1) / n_steps))
            lambd += policy_guidance_cosine_coeff * cosine_adjustment
            do_apply_guidance_this_step = jnp.logical_and(
                step_idx >= policy_guidance_delay_steps,
                step_idx < n_steps - 1,
            )
            lambd = jnp.where(
                do_apply_guidance_this_step, policy_guidance_coeff * lambd, 0.0
            )

            # --- Compute denoised trajectory for guidance ---
            guidance_traj = noised_traj
            if denoised_guidance:
                noise_pred = denoiser_state.apply_fn(
                    denoiser_state.params,
                    c_in(sigma, denoiser_hyperparams.sigma_data) * noised_traj,
                    c_noise(sigma),
                )
                guidance_traj = (
                    c_skip(sigma, denoiser_hyperparams.sigma_data) * noised_traj
                    + c_out(sigma, denoiser_hyperparams.sigma_data) * noise_pred
                )

            # --- Apply guidance ---
            action_guidance = _compute_action_guidance(guidance_traj)
            action = noised_traj[:, obs_dim : obs_dim + action_dim]
            guided_action = action + lambd * action_guidance
            noised_traj = noised_traj.at[:, obs_dim : obs_dim + action_dim].set(
                guided_action
            )

        # --- Compute first-order EDM denoise step ---
        rng, _rng = jax.random.split(rng)
        eps = denoiser_hyperparams.s_noise * jax.random.normal(_rng, noised_traj.shape)
        sigma_hat = sigma + gamma * sigma
        # JIT instability when gamma is 0
        traj_hat = jnp.where(
            gamma > 0,
            noised_traj + jnp.sqrt(sigma_hat**2 - sigma**2) * eps,
            noised_traj,
        )
        noise_pred = denoiser_state.apply_fn(
            denoiser_state.params,
            c_in(sigma_hat, denoiser_hyperparams.sigma_data) * traj_hat,
            c_noise(sigma_hat),
        )
        denoised_pred = (
            c_skip(sigma_hat, denoiser_hyperparams.sigma_data) * traj_hat
            + c_out(sigma_hat, denoiser_hyperparams.sigma_data) * noise_pred
        )
        denoised_over_sigma = (traj_hat - denoised_pred) / sigma_hat

        # --- Apply first-order EDM denoise step ---
        denoised_traj = noised_traj + (next_sigma - sigma_hat) * denoised_over_sigma

        # --- Compute EDM second-order correction ---
        if not denoiser_hyperparams.edm_first_order:
            next_noise_pred = denoiser_state.apply_fn(
                denoiser_state.params,
                c_in(next_sigma, denoiser_hyperparams.sigma_data) * denoised_traj,
                c_noise(next_sigma),
            )
            next_denoised_pred = (
                c_skip(next_sigma, denoiser_hyperparams.sigma_data) * denoised_traj
                + c_out(next_sigma, denoiser_hyperparams.sigma_data) * next_noise_pred
            )
            denoised_prime_over_sigma = (denoised_traj - next_denoised_pred) / (
                next_sigma + 1e-9
            )

            # --- Apply second-order EDM denoise step ---
            denoised_traj = jnp.where(
                next_sigma != 0,
                traj_hat
                + 0.5
                * (next_sigma - sigma_hat)
                * (denoised_over_sigma + denoised_prime_over_sigma),
                denoised_traj,
            )

        return (rng, denoised_traj, step_idx + 1), None

    # --- Denoise trajectory ---
    (rng, denoised_traj, _), _ = jax.lax.scan(
        denoise_step,
        (rng, init_noise, 0),
        (sigmas[:-1], sigmas[1:], gammas[:-1]),
    )

    # --- Construct rollout ---
    rollout, next_obs = construct_rollout(
        denoised_traj,
        denoiser_norm_stats,
        normalize_reward,
        obs_dim,
        action_dim,
    )
    return rollout, next_obs
