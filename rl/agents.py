import optax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from rl.iql import make_train_step as iql_train_step
from rl.td3_bc import make_train_step as td3_bc_train_step


DETERMINISTIC_ACTORS = ["td3_bc"]


def get_agent(args, action_dim, action_lims, obs_stats=None):
    if args.agent == "td3_bc":
        from models.td3_bc import TanhDeterministicActor, SoftQNetwork

        auxilary_networks = (
            # Target actor
            TanhDeterministicActor(
                action_dim,
                activation=args.activation,
                action_lims=action_lims,
                obs_stats=obs_stats,
            ),
            # Q network 1
            SoftQNetwork(activation=args.activation, obs_stats=obs_stats),
            # Q network 2
            SoftQNetwork(activation=args.activation, obs_stats=obs_stats),
            # Target Q network 1
            SoftQNetwork(activation=args.activation, obs_stats=obs_stats),
            # Target Q network 2
            SoftQNetwork(activation=args.activation, obs_stats=obs_stats),
        )
        return (
            TanhDeterministicActor(
                action_dim, activation=args.activation, action_lims=action_lims
            ),
            auxilary_networks,
        )
    elif args.agent == "iql":
        from models.iql import ValueFunction, VectorCritic, TanhGaussianActor

        agent_networks = {
            "train": TanhGaussianActor(
                action_dim,
                activation=args.activation,
                action_lims=action_lims,
                obs_stats=obs_stats,
            ),
            "eval": TanhGaussianActor(
                action_dim,
                activation=args.activation,
                action_lims=action_lims,
                obs_stats=obs_stats,
                eval=True,
            ),
        }
        auxilary_networks = (
            # Q network
            VectorCritic(activation=args.activation, n_critics=2, obs_stats=obs_stats),
            # Target Q network
            VectorCritic(activation=args.activation, n_critics=2, obs_stats=obs_stats),
            # Value function
            ValueFunction(activation=args.activation, obs_stats=obs_stats),
        )
        return agent_networks, auxilary_networks
    raise ValueError(f"Unknown agent {args.agent}.")


def make_train_step(args, network, aux_networks):
    if args.agent == "td3_bc":
        return td3_bc_train_step(args, network, aux_networks)
    elif args.agent == "iql":
        return iql_train_step(args, network, aux_networks)
    raise ValueError(f"Unknown agent {args.agent}.")


def get_total_num_steps(args):
    num_minibatch_updates = 1
    if args.offline:
        if args.offline_minibatch_size is not None:
            num_minibatch_updates = (
                args.offline_batch_size / args.offline_minibatch_size
            )
    else:
        num_minibatch_updates = args.online_num_minibatches
    return int(num_minibatch_updates * args.num_train_steps)


def make_lr_schedule(args):
    init_lr = args.lr
    if args.lr_schedule == "constant":
        return init_lr
    total_steps = get_total_num_steps(args)
    warmup_steps = total_steps // 10
    if args.lr_schedule == "cosine":
        return optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=total_steps,
        )
    elif args.lr_schedule == "exponential":
        return optax.warmup_exponential_decay_schedule(
            init_value=init_lr * 0.1,
            peak_value=init_lr,
            warmup_steps=warmup_steps,
            transition_steps=total_steps - warmup_steps,
            decay_rate=0.1,
        )
    raise ValueError(f"Unknown learning rate schedule {args.lr_schedule}.")


def _create_optimizer(args):
    lr_schedule = make_lr_schedule(args)
    return optax.adam(lr_schedule, eps=1e-5)


def create_agent_train_state(rng, network, args, obs_shape=None, action_dim=None):
    if obs_shape is None or hasattr(network, "NO_INPUT"):
        # Parameter
        network_params = network.init(rng)
    else:
        init_x = jnp.zeros(obs_shape)
        if action_dim is None or hasattr(network, "NO_ACTION_INPUT"):
            # Actor, critic, or actor-critic
            network_params = network.init(rng, init_x)
        else:
            # Q network
            init_action = jnp.zeros(action_dim)
            network_params = network.init(rng, init_x, init_action)
    tx = _create_optimizer(args)
    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
