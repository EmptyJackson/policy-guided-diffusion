import jax
import optax
import jax.numpy as jnp


EXP_ADV_MAX = 100.0


def make_train_step(args, network, aux_networks):
    q_network, _, value_network = aux_networks

    def _update_step(train_state, aux_train_states, traj_batch, rng):
        q_train_state, q_target_train_state, value_train_state = aux_train_states
        traj_batch = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, x.shape[-1])), traj_batch
        )

        # --- Update target networks ---
        # Done first to ensure correct target initialization
        new_target_params = jax.tree_map(
            lambda x, y: jnp.where(q_target_train_state.step == 0, x, y),
            q_train_state.params,
            optax.incremental_update(
                q_train_state.params,
                q_target_train_state.params,
                args.polyak_step_size,
            ),
        )
        q_target_train_state = q_target_train_state.replace(
            step=q_target_train_state.step + 1,
            params=new_target_params,
        )

        # --- Compute value targets ---
        q_target_preds = q_network.apply(
            q_target_train_state.params, traj_batch.obs, traj_batch.action
        )
        value_targets = jnp.min(q_target_preds, axis=-1)
        next_v = value_network.apply(value_train_state.params, traj_batch.next_obs)

        # --- Update value function ---
        def _value_loss_fn(params):
            value_pred = value_network.apply(params, traj_batch.obs)
            adv = value_targets - value_pred
            # Asymmetric L2 loss
            value_loss = jnp.abs(
                args.iql_tau - jnp.where(adv < 0.0, 1.0, 0.0)
            ) * jnp.square(adv)
            return jnp.mean(value_loss), adv

        (value_loss, adv), value_grad = jax.value_and_grad(
            _value_loss_fn, has_aux=True
        )(value_train_state.params)
        value_train_state = value_train_state.apply_gradients(grads=value_grad)

        # --- Compute q targets ---
        def _compute_q_target(transition, next_v):
            return transition.reward + args.gamma * (1 - transition.done) * next_v

        q_targets = jax.vmap(_compute_q_target)(traj_batch, next_v)

        # --- Update q functions ---
        def _q_loss_fn(params):
            # Compute loss for both critics
            q_pred = q_network.apply(params, traj_batch.obs, traj_batch.action)
            q_loss = jnp.square(q_pred - q_targets).mean()
            return q_loss

        q_loss, q_grad = jax.value_and_grad(_q_loss_fn)(q_train_state.params)
        q_train_state = q_train_state.apply_gradients(grads=q_grad)

        # --- Update actor ---
        exp_adv = jnp.exp(adv * args.iql_beta).clip(max=EXP_ADV_MAX)

        def _actor_loss_function(params):
            def _compute_loss(transition, exp_adv):
                pi = network.apply(params, transition.obs)
                bc_losses = -pi.log_prob(transition.action)
                return exp_adv * bc_losses.sum()

            actor_loss = jax.vmap(_compute_loss)(traj_batch, exp_adv)
            return actor_loss.mean()

        actor_loss, actor_grad = jax.value_and_grad(_actor_loss_function)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=actor_grad)

        loss = {
            "value_loss": value_loss,
            "q_loss": q_loss,
            "actor_loss": actor_loss,
        }
        metric = traj_batch.info
        loss = jax.tree_map(lambda x: x.mean(), loss)
        return (
            train_state,
            (q_train_state, q_target_train_state, value_train_state),
            loss,
            metric,
        )

    return _update_step
