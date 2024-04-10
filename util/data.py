import jax
import jax.numpy as jnp

from gym import spaces
from typing import NamedTuple
from functools import partial


GYM_OBS_KEYS = ["observation", "desired_goal"]


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def get_placeholder_trajectory(obs_dim, action_dim):
    batch_size = 100
    seq_len = 100
    return Transition(
        done=jnp.zeros((batch_size, seq_len, 1)),
        action=jnp.zeros((batch_size, seq_len, action_dim)),
        value=jnp.zeros((batch_size, seq_len, 1)),
        reward=jnp.zeros((batch_size, seq_len, 1)),
        log_prob=jnp.zeros((batch_size, seq_len, 1)),
        obs=jnp.zeros((batch_size, seq_len, obs_dim)),
        next_obs=jnp.zeros((batch_size, seq_len, obs_dim)),
        info=jnp.zeros((batch_size, seq_len, 1)),
    )


def stack_transitions(trajectory: Transition):
    return jnp.concatenate(
        [trajectory.obs, trajectory.action, trajectory.reward, trajectory.done], axis=-1
    )


def unstack_transitions(dataset: jnp.ndarray, obs_dim: int, action_dim: int):
    return Transition(
        obs=dataset[..., :-1, :obs_dim],
        action=dataset[..., :-1, obs_dim : obs_dim + action_dim],
        reward=dataset[..., :-1, obs_dim + action_dim : obs_dim + action_dim + 1],
        done=dataset[..., :-1, obs_dim + action_dim + 1 : obs_dim + action_dim + 2],
        next_obs=dataset[..., 1:, :obs_dim],
        info=None,
        log_prob=None,
        value=None,
    )


def dict_obs_shape(obs_shape: spaces.Dict):
    shapes = []
    for key in GYM_OBS_KEYS:
        shapes.append(obs_shape[key].shape[0])
    return tuple([sum(shapes)])


def stack_dict_obs(obs: dict):
    list_obs = []
    for key in GYM_OBS_KEYS:
        list_obs.append(obs[key])
    return jnp.concatenate(list_obs, axis=-1)


@partial(jax.vmap, in_axes=-1, out_axes=-1)
def normalise_traj(trajectories, stats=None):
    """Normalize trajectory dimension and return statistics if not provided"""
    if stats is None:
        mean = jnp.mean(trajectories)
        std = jnp.std(trajectories)
        std = jnp.where(std == 0.0, 1.0, std)
        return (trajectories - mean) / std, mean, std
    return (trajectories - stats["mean"]) / stats["std"]


@partial(jax.vmap, in_axes=-1, out_axes=-1)
def unnormalise_traj(trajectories, stats):
    return trajectories * stats["std"] + stats["mean"]


def construct_rollout(
    denoised_traj,
    denoiser_norm_stats,
    normalize_reward,
    obs_dim,
    action_dim,
):
    rollout = unstack_transitions(denoised_traj, obs_dim, action_dim)
    obs = unnormalise_traj(rollout.obs, denoiser_norm_stats["obs"])
    action = unnormalise_traj(rollout.action, denoiser_norm_stats["action"])
    action = jnp.tanh(action)
    reward = rollout.reward
    if not normalize_reward:
        unnormalise_traj(rollout.reward, denoiser_norm_stats["reward"])
    done = unnormalise_traj(rollout.done, denoiser_norm_stats["done"])
    done = jnp.greater(done, 0.5).astype(jnp.float32)
    return (
        Transition(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            next_obs=unnormalise_traj(rollout.next_obs, denoiser_norm_stats["obs"]),
            value=None,
            log_prob=None,
            info=None,
        ),
        obs[-1],
    )
