import jax
import jax.numpy as jnp
import gym
import d4rl

from functools import partial

from util import *


def load_dataset(args, normalize, val_split=0.0):
    """Load and normalize train and validation datasets"""
    trajs, val_trajs = _load_dataset(args, val_split=val_split)
    if normalize:
        trajs, trajectory_norm_stats = _normalize_dataset(
            trajs._replace(action=jnp.arctanh(jnp.clip(trajs.action, -0.999, 0.999)))
        )
        if val_trajs is not None:
            val_trajs = _normalize_from_stats(
                val_trajs._replace(
                    action=jnp.arctanh(jnp.clip(val_trajs.action, -0.999, 0.999))
                ),
                trajectory_norm_stats,
            )
    obs_dim, num_actions = trajs.obs.shape[-1], trajs.action.shape[-1]
    if normalize:
        return trajs, val_trajs, trajectory_norm_stats, (obs_dim, num_actions)
    return trajs, val_trajs, (obs_dim, num_actions)


def _load_d4rl_data(args):
    """Load D4RL dataset in Jax Numpy format, split on done flags."""
    # --- Load data and convert to Jax Numpy ---
    dataset = gym.make(args.dataset_name).get_dataset()
    trajs = {
        attr: dataset[attr][:-1]
        for attr in ["observations", "actions", "rewards", "terminals", "timeouts"]
    }
    trajs["next_observations"] = dataset["observations"][1:]
    trajs["done"] = jnp.logical_or(dataset["terminals"][:-1], dataset["timeouts"][:-1])
    trajs = jax.tree_map(jnp.array, trajs)

    # --- Split data on terminal or timeout flags ---
    split_idxs = jnp.argwhere(trajs["done"]).squeeze() + 1
    # Omit final index if present
    if split_idxs[-1] == len(trajs["done"]):
        split_idxs = split_idxs[:-1]
    trajs = jax.tree_map(lambda x: jnp.array_split(x, split_idxs), trajs)

    # --- Return list of episode dicts ---
    return [{k: v[i] for k, v in trajs.items()} for i in range(len(split_idxs) + 1)]


def _load_dataset(args, val_split=0.0):
    """
    Loads flattened D4RL dataset.

    Episodes are concatenated together,
    then split into args.trajectory_length around done flags.
    """
    print(f"Loading D4RL dataset {args.dataset_name}", end="...")

    # --- Load training and validation episodes ---
    eps = _load_d4rl_data(args)
    if val_split > 0.0:
        num_val_eps = int(val_split * len(eps))
        print(
            f"found {len(eps)} episodes, splitting off {num_val_eps} for validation.",
        )
        assert (
            num_val_eps > 0
        ), f"Val split {val_split} too small given {len(eps)} episodes"
        val_ep_idxs = jax.random.choice(
            jax.random.PRNGKey(args.seed),
            len(eps),
            shape=(num_val_eps,),
            replace=False,
        )
        val_eps = [eps[i] for i in val_ep_idxs]
        eps = [ep for i, ep in enumerate(eps) if i not in val_ep_idxs]
    else:
        print(f"found {len(eps)} episodes, no validation set.")

    def _assemble_dataset(eps):
        """
        Assemble subtrajectory dataset from list of episodes.

        Subtrajectories have length args.trajectory_length,
        with args.dataset_stride stride across dataset.

        Subtrajectories never reset at intermediate steps, or timeout at
        any step (done flag corresponds to terminal only).
        """
        if args.trajectory_length > 1:
            # --- Concatenate episodes and find global episode start indices ---
            print("Assembling dataset", end="...")
            flat_done = jnp.concatenate([ep["done"] for ep in eps], axis=0)
            done_idxs = jnp.argwhere(flat_done).squeeze(axis=-1)
            if done_idxs[-1] == len(flat_done) - 1:
                done_idxs = done_idxs[:-1]
            init_idxs = jnp.concatenate([jnp.zeros(1), done_idxs + 1], axis=0)

            # --- Compute subtrajectory indices without intermediate episode resets ---
            any_done = jax.jit(partial(jnp.convolve, mode="valid"))(
                a=jnp.ones(args.trajectory_length - 1), v=flat_done[:-1]
            )
            valid_start_idxs = jnp.argwhere(any_done == 0).squeeze(axis=-1)

            # --- Compute subtrajecories ending with terminal or timeout ---
            flat_term = jnp.concatenate([ep["terminals"] for ep in eps], axis=0)
            term_idxs = jnp.argwhere(flat_term).squeeze(axis=-1)
            flat_timeout = jnp.concatenate([ep["timeouts"] for ep in eps], axis=0)
            timeout_idxs = jnp.argwhere(flat_timeout).squeeze(axis=-1)
            print(
                f"{len(term_idxs)} terminal, {len(timeout_idxs)} timeout flags found",
                end="...",
            )
            term_idxs -= args.trajectory_length - 1
            timeout_idxs -= args.trajectory_length - 1

            # --- Compute subtrajectory indices ---
            # Add strided subtrajectories
            start_idxs = set(valid_start_idxs[:: args.dataset_stride].tolist())
            # Add the start and end (final step terminal) of episodes
            start_idxs |= set(valid_start_idxs.tolist()) & set(term_idxs.tolist())
            start_idxs |= set(valid_start_idxs.tolist()) & set(init_idxs.tolist())
            # Remove subtrajectories ending in timeout
            start_idxs -= set(timeout_idxs.tolist())
            # Compute index array from list of start positions
            start_idxs = jnp.array(list(start_idxs), dtype=jnp.int32)
            subtraj_idxs = jax.jit(
                jax.vmap(lambda x: jnp.arange(args.trajectory_length) + x)
            )(start_idxs)
        else:
            # --- Remove timeout transitions ---
            flat_timeout = jnp.concatenate([ep["timeouts"] for ep in eps], axis=0)
            subtraj_idxs = jnp.argwhere(~flat_timeout).squeeze(axis=-1)

        # --- Construct subtrajectories from indices ---
        def _construct_tensor(data, add_singleton=False):
            # --- Construct Jax Numpy array from subtrajectory indices ---
            ret = jnp.concatenate(data, axis=0)
            ret = jnp.take(ret, subtraj_idxs, axis=0)
            if add_singleton:
                # Add singleton dimension
                return jnp.expand_dims(ret, axis=-1)
            return ret

        trajectories = Transition(
            obs=_construct_tensor([ep["observations"] for ep in eps]),
            action=_construct_tensor([ep["actions"] for ep in eps]),
            reward=_construct_tensor([ep["rewards"] for ep in eps], add_singleton=True),
            next_obs=_construct_tensor([ep["next_observations"] for ep in eps]),
            done=_construct_tensor([ep["terminals"] for ep in eps], add_singleton=True),
            value=None,
            log_prob=None,
            info=None,
        )
        print(f"done ({len(subtraj_idxs)} subtrajectories constructed).")
        print(f"Number of terminals: {jnp.sum(trajectories.done)}")
        assert ~jnp.any(
            trajectories.done[:, :-1]
        ), "Done flags in the middle of subtrajectory"
        return trajectories

    # --- Return assembled training and validation datasets ---
    return (
        _assemble_dataset(eps),
        _assemble_dataset(val_eps) if val_split > 0.0 else None,
    )


def _normalize_dataset(trajs):
    """Normalize observations, actions, rewards and done flags"""
    obs, obs_norm_mean, obs_norm_std = normalise_traj(trajs.obs)
    obs_stats = {"mean": obs_norm_mean, "std": obs_norm_std}
    next_obs = normalise_traj(trajs.next_obs, obs_stats)
    action, action_norm_mean, action_norm_std = normalise_traj(trajs.action)
    reward, reward_norm_mean, reward_norm_std = normalise_traj(trajs.reward)
    done, done_norm_mean, done_norm_std = normalise_traj(trajs.done)
    trajectory_norm_stats = {
        "obs": obs_stats,
        "action": {"mean": action_norm_mean, "std": action_norm_std},
        "reward": {"mean": reward_norm_mean, "std": reward_norm_std},
        "done": {"mean": done_norm_mean, "std": done_norm_std},
    }
    return (
        trajs._replace(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            next_obs=next_obs,
        ),
        trajectory_norm_stats,
    )


def _normalize_from_stats(trajs, stats):
    """Normalize observations, actions, rewards and done flags with given statistics"""
    return trajs._replace(
        obs=normalise_traj(trajs.obs, stats["obs"]),
        next_obs=normalise_traj(trajs.next_obs, stats["obs"]),
        action=normalise_traj(trajs.action, stats["action"]),
        reward=normalise_traj(trajs.reward, stats["reward"]),
        done=normalise_traj(trajs.done, stats["done"]),
    )
