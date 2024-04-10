import sys
import jax
import jax.numpy as jnp

from flax.training import orbax_utils
from jax.config import config as jax_config
from orbax.checkpoint import PyTreeCheckpointer

from util import *
from environments.dataset import load_dataset
from diffusion.diffusion import (
    make_train_step,
    create_denoiser_train_state,
)


def make_train(args, dataset, val_dataset, num_epochs):
    diffusion_train_fn = make_train_step(args)

    def train(rng, denoiser_state, ema_denoiser_state):
        # --- TRAIN LOOP ---
        def _epoch_train_step(runner_state, _):
            denoiser_state, ema_denoiser_state, rng = runner_state

            def _batch_train_step(runner_state, batch):
                # --- Update model on batch ---
                denoiser_state, ema_denoiser_state, rng = runner_state
                rng, _rng = jax.random.split(rng)
                denoiser_state, loss = diffusion_train_fn(_rng, batch, denoiser_state)

                # --- Update EMA (exponential moving average) model ---
                new_params = ema_update(args, denoiser_state, ema_denoiser_state)
                ema_denoiser_state = ema_denoiser_state.replace(params=new_params)
                return (denoiser_state, ema_denoiser_state, rng), jnp.mean(loss)

            # --- Construct and iterate over dataset mini-batches ---
            rng, _rng = jax.random.split(rng)
            batched_dataset = shuffle_and_batch_dataset(_rng, dataset, args.batch_size)
            (denoiser_state, ema_denoiser_state, rng), batch_losses = jax.lax.scan(
                _batch_train_step,
                (denoiser_state, ema_denoiser_state, rng),
                batched_dataset,
            )
            return (denoiser_state, ema_denoiser_state, rng), jnp.mean(batch_losses)

        # --- Iterate over dataset epochs ---
        (denoiser_state, ema_denoiser_state, _), train_losses = jax.lax.scan(
            _epoch_train_step,
            (denoiser_state, ema_denoiser_state, rng),
            None,
            length=num_epochs,
        )

        # --- Compute validation loss ---
        def _batch_eval(rng, batch):
            rng, _rng = jax.random.split(rng)
            _, loss = diffusion_train_fn(_rng, batch, ema_denoiser_state)
            return rng, jnp.mean(loss)

        rng, _rng = jax.random.split(rng)
        batched_val_dataset = shuffle_and_batch_dataset(
            _rng, val_dataset, args.batch_size
        )
        _, val_losses = jax.lax.scan(_batch_eval, rng, batched_val_dataset)
        return denoiser_state, ema_denoiser_state, train_losses, jnp.mean(val_losses)

    return train


def train_offline_diffusion(args):
    rng = jax.random.PRNGKey(args.seed)

    # --- Construct training and validation datasets ---
    trajs, val_trajs, trajectory_norm_stats, (obs_dim, action_dim) = load_dataset(
        args, normalize=True, val_split=args.val_ratio
    )
    dataset = stack_transitions(trajs)
    val_dataset = stack_transitions(val_trajs)
    if args.log:
        wandb.config.update({"norm_stats": trajectory_norm_stats})

    # --- Create denoiser state (and EMA copy) ---
    rng, _rng = jax.random.split(rng)
    denoiser_state = create_denoiser_train_state(
        _rng, obs_dim, action_dim, args, dataset.shape[0]
    )
    ema_denoiser_state = jax.tree_map(jnp.copy, denoiser_state)

    # --- TRAIN + LOG LOOP ---
    train_fn = jax.jit(make_train(args, dataset, val_dataset, args.eval_rate))
    for start_epoch_n in range(1, args.num_epochs + 1, args.eval_rate):
        # --- Train for eval_rate epochs ---
        rng, _rng = jax.random.split(rng)
        denoiser_state, ema_denoiser_state, train_losses, val_loss = train_fn(
            _rng, denoiser_state, ema_denoiser_state
        )

        # --- Log metrics ---
        final_epoch_n = start_epoch_n + args.eval_rate
        print(
            f"Epoch: {final_epoch_n}, Train: {train_losses[-1]:0.4f}, Val: {val_loss:0.4f}"
        )
        if args.log:
            for epoch_idx in range(start_epoch_n, final_epoch_n - 1):
                log(
                    {
                        "epoch": epoch_idx,
                        "train_loss": train_losses[epoch_idx - start_epoch_n],
                    }
                )
            log(
                {
                    "epoch": final_epoch_n,
                    "step": denoiser_state.step,
                    "train_loss": train_losses[-1],
                    "validation_loss": val_loss,
                }
            )

    # --- Save checkpoint ---
    if args.log and args.save_checkpoint:
        ckptr = PyTreeCheckpointer()
        ckptr.save(
            os.path.join(wandb.run.dir, CHECKPOINT_DIR),
            ema_denoiser_state,
            save_args=orbax_utils.save_args_from_target(ema_denoiser_state),
        )


def main(cmd_args=sys.argv[1:]):
    # --- Parse arguments and initialize logging ---
    args = parse_diffusion_args(cmd_args)
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type="train_diffusion",
        )
    debug = args.debug
    debug_nans = args.debug_nans
    if debug_nans:
        jax_config.update("jax_debug_nans", True)

    # --- Launch training ---
    if debug:
        with jax.disable_jit():
            return train_offline_diffusion(args)
    else:
        return train_offline_diffusion(args)


if __name__ == "__main__":
    main()
