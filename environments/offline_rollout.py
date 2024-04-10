import jax

from environments.dataset import load_dataset
from util import *


class DatasetRolloutGenerator:
    """Parent class for rollout generators that use a dataset"""

    def __init__(self, dataset, batch_size):
        self._dataset = dataset
        self._num_transitions = self._dataset.obs.shape[0]
        self._batch_size = batch_size

        def _get_batch(data, rng):
            permutation = jax.random.choice(
                rng,
                jnp.arange(self._num_transitions),
                shape=(self._batch_size,),
                replace=False,
            )
            # Sample transitions from dataset
            batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), data
            )
            # Reshape batch to conform with online rollout shape
            return jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (x.shape[0], 1, *x.shape[1:])), batch
            )

        self.batch_fn = jax.jit(_get_batch)

    def batch_rollout(self, rng):
        return self.batch_fn(self._dataset, rng)


class OfflineRolloutGenerator(DatasetRolloutGenerator):
    def __init__(
        self,
        args,
        obs_shape,
        action_dim,
        discrete_actions,
        action_lims,
        num_env_steps,
        agent_apply_fn=None,
        batch_size=None,
    ):
        self.num_env_steps = num_env_steps
        self.agent_apply_fn = agent_apply_fn
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.action_lims = action_lims

        args.trajectory_length = 2
        args.dataset_stride = 2
        self.trajectory_length = 2
        transitions = load_dataset(
            args, normalize=False, normalize_reward=args.normalize_reward
        )[0]
        # Flatten dataset
        transitions = jax.tree_map(lambda x: x.reshape((-1, x.shape[-1])), transitions)
        self.obs_stats = {
            "mean": transitions.obs.mean(axis=0),
            "std": transitions.obs.std(axis=0),
        }
        if batch_size is None:
            batch_size = args.offline_batch_size
        super().__init__(transitions, batch_size)

    def set_apply_fn(self, agent_apply_fn):
        self.agent_apply_fn = agent_apply_fn
