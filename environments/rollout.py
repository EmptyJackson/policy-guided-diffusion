import jax
import numpy as onp
import gym
import d4rl
from typing import Optional

from util import *


def get_gym_env(dataset_name: str, num_env_workers: int):
    """Returns a Gym environment, matching the D4RL dataset configuration."""
    env = gym.vector.make(dataset_name, num_envs=num_env_workers)
    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    env = gym.wrappers.RecordEpisodeStatistics(env)
    has_dict_obs = isinstance(env.single_observation_space, gym.spaces.Dict)
    return env, has_dict_obs, max_episode_steps


class GymRolloutWrapper:
    def __init__(
        self,
        env_name: str,
        num_env_steps: Optional[int] = None,
        agent_apply_fn: Optional[callable] = None,
        num_env_workers: Optional[int] = None,
    ):
        self.env_name = env_name
        self.env, self.convert_dict_obs, max_episode_steps = get_gym_env(
            env_name, num_env_workers
        )
        self.agent_apply_fn = agent_apply_fn
        if num_env_steps is None:
            self.num_env_steps = max_episode_steps
        else:
            self.num_env_steps = num_env_steps

    def batch_reset(self, rng, num_env_workers):
        """Reset a single environment over a batch of seeds."""
        seeds = jax.random.split(rng, num_env_workers)
        reset_obs = self.env.reset(seed=[int(i[0]) for i in seeds])
        if self.convert_dict_obs:
            reset_obs = stack_dict_obs(reset_obs)
        return reset_obs

    def batch_rollout(self, rng, agent_state, last_obs):
        """Evaluate an agent on a single environment over a batch of seeds and environment states."""

        @jax.jit
        @jax.vmap
        def _policy_step(rng, obs):
            # --- Compute next action for a single state ---
            pi = self.agent_apply_fn(agent_state.params, obs)
            rng, _rng = jax.random.split(rng)
            action, log_prob = pi.sample_and_log_prob(seed=_rng)
            action = jnp.nan_to_num(action)
            # Sum action dimension log probabilities
            log_prob = log_prob.sum(axis=-1)
            return rng, action, log_prob

        transition_list = []
        num_env_workers = last_obs.shape[0]
        rng = jax.random.split(rng, num_env_workers)
        returned = [False for _ in range(num_env_workers)]
        for _ in range(self.num_env_steps):
            # --- Take step in environment ---
            rng, action, log_prob = _policy_step(rng, jnp.array(last_obs))
            obs, reward, done, info = self.env.step(onp.array(action))
            if self.convert_dict_obs:
                obs = stack_dict_obs(obs)

            # --- Track cumulative reward ---
            new_returned = []
            returned_episode_returns = []
            for worker_id, worker_returned in enumerate(returned):
                if "episode" in info[worker_id].keys() and not worker_returned:
                    returned_episode_returns.append(info[worker_id]["episode"]["r"])
                    new_returned.append(True)
                else:
                    returned_episode_returns.append(jnp.nan)
                    new_returned.append(worker_returned)
            returned = new_returned
            returned_episode_returns = jnp.array(returned_episode_returns)
            info = {
                "returned_episode_returns": returned_episode_returns,
                "returned_episode_scores": d4rl.get_normalized_score(
                    self.env_name, returned_episode_returns
                )
                * 100.0,
            }

            # --- Construct transition ---
            transition_list.append(
                Transition(
                    obs=last_obs,
                    action=action,
                    reward=jnp.expand_dims(reward, axis=-1),
                    done=jnp.expand_dims(done, axis=-1),
                    next_obs=obs,
                    log_prob=jnp.expand_dims(log_prob, axis=-1),
                    value=None,
                    info=info,
                )
            )
            last_obs = obs
        return tree_stack(transition_list)

    @property
    def obs_shape(self):
        """Get the shape of the observation."""
        if self.convert_dict_obs:
            return dict_obs_shape(self.env.single_observation_space)
        return self.env.single_observation_space.shape

    @property
    def action_dim(self):
        """Get the dimension of the action space."""
        return self.env.single_action_space.shape[0]

    @property
    def action_lims(self):
        """Get the action limits for the environment."""
        return (
            self.env.single_action_space.low[0],
            self.env.single_action_space.high[0],
        )

    def set_apply_fn(self, agent_apply_fn):
        self.agent_apply_fn = agent_apply_fn
