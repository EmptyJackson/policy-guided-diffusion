import jax


def eval_agents(rng, eval_rollout_gen, train_state, num_env_workers):
    rng, _rng = jax.random.split(rng)
    last_obs = eval_rollout_gen.batch_reset(_rng, num_env_workers)
    traj_batch = eval_rollout_gen.batch_rollout(rng, train_state, last_obs)
    return traj_batch
