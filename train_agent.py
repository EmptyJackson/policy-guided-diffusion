import jax
import sys

import jax.numpy as jnp
from jax.config import config as jax_config

from util import *
from rl.agents import create_agent_train_state, make_train_step, get_agent
from environments.rollout import GymRolloutWrapper
from environments.offline_rollout import OfflineRolloutGenerator
from diffusion.rollout_generator import MixedRolloutGenerator


def make_train(args):
    """Makes agent train function."""

    def _init_env(rng):
        # --- Initialize real environment for evaluation ---
        env = GymRolloutWrapper(args.env_name, num_env_workers=args.num_env_workers)
        if args.synthetic_experience:
            # --- Initialize mixed (synthetic + dataset) sampler ---
            rng, _rng = jax.random.split(rng)
            rollout_gen = MixedRolloutGenerator(
                _rng,
                args,
                env.obs_shape,
                env.action_dim,
                env.action_lims,
                args.num_rollout_steps,
            )
        else:
            # --- Initialize dataset sampler ---
            rollout_gen = OfflineRolloutGenerator(
                args,
                env.obs_shape,
                env.action_dim,
                env.action_lims,
                args.num_rollout_steps,
            )
        return rollout_gen, env

    def _init_agent(rng, rollout_gen):
        # --- Get agent networks ---
        network, aux_networks = get_agent(
            args,
            rollout_gen.action_dim,
            rollout_gen.action_lims,
            obs_stats=rollout_gen.obs_stats if args.normalize_obs else None,
        )
        if isinstance(network, dict):
            eval_apply_fn = network["eval"].apply
            network = network["train"]
        else:
            eval_apply_fn = network.apply

        # --- Create agent train states ---
        rng, _rng = jax.random.split(rng)
        train_state = create_agent_train_state(
            _rng, network, args, rollout_gen.obs_shape
        )
        if aux_networks is None:
            return train_state, None, network, aux_networks, eval_apply_fn

        # --- Create auxiliary train states ---
        aux_train_states = []
        for net in aux_networks:
            rng, _rng = jax.random.split(rng)
            ts = create_agent_train_state(
                _rng, net, args, rollout_gen.obs_shape, rollout_gen.action_dim
            )
            aux_train_states.append(ts)
        aux_train_states = tuple(aux_train_states)
        return train_state, aux_train_states, network, aux_networks, eval_apply_fn

    def train(rng):
        # --- Initialize environment ---
        rng, _rng = jax.random.split(rng)
        rollout_gen, env = _init_env(_rng)

        # --- Initialize agent (policy + value) and auxiliary networks ---
        rng, _rng = jax.random.split(rng)
        (
            train_state,
            aux_train_states,
            network,
            aux_networks,
            eval_apply_fn,
        ) = _init_agent(_rng, rollout_gen)
        rollout_gen.set_apply_fn(jax.jit(train_state.apply_fn))
        env.set_apply_fn(jax.jit(eval_apply_fn))
        _agent_train_step_fn = jax.jit(make_train_step(args, network, aux_networks))

        losses, metrics = [], []
        for step_idx in range(args.num_train_steps):
            # --- Sample batch and update agent ---
            rng, _rng = jax.random.split(rng)
            traj_batch = rollout_gen.batch_rollout(_rng)
            rng, _rng = jax.random.split(rng)
            train_state, aux_train_states, loss, metric = _agent_train_step_fn(
                train_state, aux_train_states, traj_batch, _rng
            )
            losses.append(loss)

            # --- Evaluate agent ---
            if step_idx % args.eval_rate == 0:
                rng, _rng = jax.random.split(rng)
                eval_traj_batch = eval_agents(
                    rng,
                    env,
                    train_state,
                    args.num_env_workers,
                )
                info = eval_traj_batch.info
                metric = {
                    "num_updates": train_state.step,
                    "returned_episode_returns": jnp.nanmean(
                        info["returned_episode_returns"]
                    ),
                    "returned_episode_scores": jnp.nanmean(
                        info["returned_episode_scores"]
                    ),
                }
                metrics.append(metric)

            # --- Regenerate synthetic dataset (if not finished) ---
            if (
                args.synthetic_experience
                and step_idx % args.synth_dataset_lifetime == 0
                and step_idx != 0
            ):
                rng, _rng = jax.random.split(rng)
                rollout_gen.update_synthetic_dataset(_rng, train_state.params)
        return metrics, losses

    return train


def train_agents(args):
    rng = jax.random.PRNGKey(args.seed)

    # --- Train agent and log metrics ---
    train_fn = make_train(args)
    metric, loss = train_fn(rng)
    if args.log:
        # --- Compute mean return and score per step ---
        returns = [met["returned_episode_returns"] for met in metric]
        scores = [met["returned_episode_scores"] for met in metric]
        num_updates = [met["num_updates"] for met in metric]

        # --- Subsample steps for logging ---
        if len(returns) > MAX_LOG_STEPS:
            steps = jnp.linspace(0, len(returns), MAX_LOG_STEPS, dtype=jnp.int32)
        else:
            steps = jnp.arange(len(returns))

        # --- Log step metrics ---
        for step in steps:
            # Log nearest step with return value
            log(
                {
                    "episode_return": returns[step],
                    "episode_score": scores[step],
                    "step": step,
                    "num_updates": num_updates[step],
                    **loss[step * args.eval_rate],
                }
            )


def main(cmd_args=sys.argv[1:]):
    # --- Parse arguments and initialize logging ---
    args = parse_agent_args(cmd_args)
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type="train_agent",
        )

    debug = args.debug
    debug_nans = args.debug_nans

    if debug_nans:
        jax_config.update("jax_debug_nans", True)

    # --- Launch training ---
    if debug:
        with jax.disable_jit():
            return train_agents(args)
    else:
        return train_agents(args)


if __name__ == "__main__":
    main()
