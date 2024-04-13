import jax


def eval_agents(rng, env, train_state, num_env_workers):
    rng, _rng = jax.random.split(rng)
    last_obs = env.batch_reset(_rng, num_env_workers)
    traj_batch = env.batch_rollout(rng, train_state, last_obs)
    return traj_batch
