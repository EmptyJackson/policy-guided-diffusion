import jax
import jax.numpy as jnp


def scan_final_only(f, init, xs, length=None, reverse=False, unroll=1):
    carry, stacked_y = jax.lax.scan(
        f, init, xs, length=length, reverse=reverse, unroll=unroll
    )
    # Return output from only the final iteration
    return carry, jax.tree_map(lambda x: x[-1], stacked_y)


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def scan_no_jit(f, init, xs, length=None, reverse=False, unroll=1):
    """Native Python implementation of JAX scan (without JIT compilaton)."""
    if xs is None:
        xs = [None] * length
    if reverse:
        xs = xs[::-1]
    carry = init
    out_list = []
    for x in xs:
        carry, out = f(carry, x)
        out_list.append(out)
    return carry, tree_stack(out_list)


def ema_update(args, denoiser_state, ema_denoiser_state):
    ema_updated_params = jax.tree_map(
        lambda x, y: args.ema_decay * x + (1 - args.ema_decay) * y,
        ema_denoiser_state.params,
        denoiser_state.params,
    )
    return jax.tree_map(
        lambda x, y: jnp.where(denoiser_state.step % args.ema_update_every == 0, x, y),
        ema_updated_params,
        ema_denoiser_state.params,
    )


def shuffle_and_batch_dataset(rng, dataset, batch_size):
    """Shuffles and batches dataset (with extra samples truncated)"""
    assert dataset.shape[0] >= batch_size, "Dataset smaller than batch"
    set_shuffled = jax.random.permutation(rng, dataset)
    return set_shuffled[dataset.shape[0] % batch_size :].reshape(
        (-1, batch_size, *dataset.shape[1:])
    )
