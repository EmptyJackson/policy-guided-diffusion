import jax
import optax
import jax.numpy as jnp


def make_train_step(args, network, aux_networks):
    _, q_network, _, _, _ = aux_networks

    def _update_step(train_state, aux_train_states, traj_batch, rng):
        (
            actor_target_state,
            q1_state,
            q2_state,
            q1_target_state,
            q2_target_state,
        ) = aux_train_states
        traj_batch = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, x.shape[-1])), traj_batch
        )

        # --- Update target networks ---
        def _update_target(state, target_state):
            # Done first to ensure correct target initialization
            new_target_params = jax.tree_map(
                lambda x, y: jnp.where(target_state.step == 0, x, y),
                state.params,
                optax.incremental_update(
                    state.params,
                    target_state.params,
                    args.polyak_step_size,
                ),
            )
            return target_state.replace(
                step=target_state.step + 1,
                params=new_target_params,
            )

        q1_target_state = _update_target(q1_state, q1_target_state)
        q2_target_state = _update_target(q2_state, q2_target_state)
        actor_target_state = _update_target(train_state, actor_target_state)

        # --- Update actor ---
        def _actor_loss_function(params, rng):
            def _transition_loss(rng, transition):
                pi = network.apply(params, transition.obs)
                sampled_action = pi.sample(seed=rng)
                q = q_network.apply(q1_state.params, transition.obs, sampled_action)
                bc_loss = jnp.square(sampled_action - transition.action).mean()
                return q, bc_loss

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, len(traj_batch.reward))
            q, bc_loss = jax.vmap(_transition_loss)(_rng, traj_batch)
            lmbda = args.td3_alpha / (jnp.abs(q).mean() + 1e-7)
            lmbda = jax.lax.stop_gradient(lmbda)
            actor_loss = (-lmbda * q.mean()) + bc_loss.mean()
            return actor_loss.mean(), (q.mean(), lmbda.mean(), bc_loss.mean())

        rng, _rng = jax.random.split(rng)
        (actor_loss, (q_mean, lmbda, bc_loss)), actor_grad = jax.value_and_grad(
            _actor_loss_function, has_aux=True
        )(train_state.params, _rng)
        train_state = train_state.apply_gradients(grads=actor_grad)

        def _update_critics(runner_state, _):
            rng, q1_state, q2_state = runner_state

            # --- Compute targets ---
            def _compute_target(rng, transition):
                next_pi = network.apply(
                    actor_target_state.params, transition.next_obs
                )
                rng, _rng = jax.random.split(rng)
                next_action = next_pi.sample(seed=_rng)
                rng, _rng = jax.random.split(rng)
                rand_action = (
                    jax.random.normal(_rng, shape=next_action.shape) * args.policy_noise
                )
                rand_action = jnp.clip(
                    rand_action, a_min=-args.noise_clip, a_max=args.noise_clip
                )
                next_action = jnp.clip(
                    next_action + rand_action, a_min=-args.a_max, a_max=args.a_max
                )

                # Minimum of the target Q-values
                target_q1 = q_network.apply(
                    q1_target_state.params, transition.next_obs, next_action
                )
                target_q2 = q_network.apply(
                    q2_target_state.params, transition.next_obs, next_action
                )
                next_q_value = jnp.minimum(target_q1, target_q2)
                assert next_q_value.shape == transition.reward.shape
                return (
                    transition.reward
                    + args.gamma * (1 - transition.done) * next_q_value
                )

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, len(traj_batch.reward))
            targets = jax.vmap(_compute_target)(_rng, traj_batch)

            # --- Update critics ---
            @jax.value_and_grad
            def _q_loss_fn(params):
                q_pred = q_network.apply(params, traj_batch.obs, traj_batch.action)
                assert q_pred.shape == targets.shape
                return jnp.square(q_pred - targets).mean()

            q1_loss, q1_grad = _q_loss_fn(q1_state.params)
            q1_state = q1_state.apply_gradients(grads=q1_grad)
            q2_loss, q2_grad = _q_loss_fn(q2_state.params)
            q2_state = q2_state.apply_gradients(grads=q2_grad)
            return (rng, q1_state, q2_state), (q1_loss, q2_loss)

        (rng, q1_state, q2_state), (q1_loss, q2_loss) = jax.lax.scan(
            _update_critics,
            (rng, q1_state, q2_state),
            None,
            length=args.num_critic_updates_per_step,
        )

        loss = {
            "q1_loss": q1_loss.mean(),
            "q2_loss": q2_loss.mean(),
            "actor_loss": actor_loss,
            "q_mean": q_mean,
            "lmbda": lmbda,
            "bc_loss": bc_loss,
        }
        metric = traj_batch.info
        return (
            train_state,
            (actor_target_state, q1_state, q2_state, q1_target_state, q2_target_state),
            loss,
            metric,
        )

    return _update_step
