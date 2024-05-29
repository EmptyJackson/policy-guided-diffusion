import jax
from argparse import Namespace
from orbax.checkpoint import PyTreeCheckpointer
from functools import partial

from diffusion.diffusion import create_denoiser_train_state, make_sample_fn
from environments.offline_rollout import (
    DatasetRolloutGenerator,
    OfflineRolloutGenerator,
)
from rl.agents import DETERMINISTIC_ACTORS
from util import *


class SyntheticRolloutGenerator(DatasetRolloutGenerator):
    def __init__(
        self,
        rng,
        args,
        obs_shape,
        action_dim,
        action_lims,
        num_env_steps,
        agent_apply_fn=None,
        batch_size=None,
    ):
        self.num_env_steps = num_env_steps
        self.agent_apply_fn = agent_apply_fn
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_lims = action_lims
        self.policy_guidance_coeff = args.policy_guidance_coeff
        self.policy_guidance_cosine_coeff = args.policy_guidance_cosine_coeff
        self.num_synth_workers = args.num_synth_workers
        self.num_synth_rollouts = args.num_synth_rollouts

        if not args.denoiser_checkpoint:
            raise ValueError(
                "Must specify generator checkpoint to use synthetic experience"
            )
        self._restore_diffusion_model(args)
        self.obs_stats = self.denoiser_norm_stats["obs"]
        det_guidance = args.agent in DETERMINISTIC_ACTORS
        self.diffusion_sample_fn = partial(
            make_sample_fn(
                self.denoiser_config,
                args.normalize_action_guidance,
                args.denoised_guidance,
                det_guidance,
            ),
            denoiser_state=self.denoiser_state,
            seq_len=self.num_env_steps + 1,
            obs_dim=self.obs_shape[0],
            action_dim=self.action_dim,
            denoiser_norm_stats=self.denoiser_norm_stats,
            policy_guidance_coeff=self.policy_guidance_coeff,
            policy_guidance_cosine_coeff=self.policy_guidance_cosine_coeff,
        )

        # Generate unguided synthetic dataset
        self.update_synthetic_dataset(rng, None)
        if batch_size is None:
            batch_size = args.batch_size
        super().__init__(self._dataset, batch_size)

    def set_apply_fn(self, agent_apply_fn):
        self.agent_apply_fn = agent_apply_fn

    def _generate_single_rollout(self, rng, agent_params):
        return self.diffusion_sample_fn(
            rng=rng, agent_params=agent_params, agent_apply_fn=self.agent_apply_fn
        )

    def update_synthetic_dataset(self, rng, agent_params=None):
        # Regenerate synthetic dataset from the current agent state
        synth_rollouts = []
        batch_rollout_fn = jax.jit(
            jax.vmap(self._generate_single_rollout, in_axes=(0, None))
        )
        for _ in range(self.num_synth_rollouts):
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, self.num_synth_workers)
            synth_rollouts.append(batch_rollout_fn(_rng, agent_params))
        # Stack and flatten rollouts
        self._dataset = jax.jit(
            lambda x: jax.tree_map(
                lambda y: y.reshape((-1, y.shape[-1])), tree_stack(x)
            )
        )(synth_rollouts)

    def _restore_diffusion_model(self, args):
        # Download checkpoint from wandb
        api = wandb.Api()
        ckpt_run = api.run(
            f"{args.wandb_team}/{args.wandb_project}/{args.denoiser_checkpoint}"
        )
        for file in ckpt_run.files():
            file.download(
                root=os.path.join("tmp", args.denoiser_checkpoint), exist_ok=True
            )
        # Create placeholder train state
        ckpt_dict = ckpt_run.config
        self.denoiser_config = Namespace(**ckpt_dict)
        if args.diffusion_timesteps is not None:
            self.denoiser_config.diffusion_timesteps = args.diffusion_timesteps
        placeholder_train_state = create_denoiser_train_state(
            jax.random.PRNGKey(0),
            self.obs_shape[0],
            self.action_dim,
            self.denoiser_config,
            10000,  # Random dataset length to create LR schedule
        )
        # Restore checkpoint into placeholder train state
        ckptr = PyTreeCheckpointer()
        self.denoiser_state = ckptr.restore(
            os.path.join("tmp", args.denoiser_checkpoint, CHECKPOINT_DIR),
            item=placeholder_train_state,
        )
        # Restore normalization statistics
        # Temporary hack, some of the stats are stored as strings
        def conv_str(s):
            s = s.replace("\n", "")
            s = s.replace("[", "")
            s = s.replace("]", "")
            return [float(x) for x in s.split(" ") if x != ""]

        ckpt_dict["norm_stats"] = {
            k: {k1: v if not isinstance(v, str) else conv_str(v) for k1, v in x.items()}
            for k, x in ckpt_dict["norm_stats"].items()
        }
        self.denoiser_norm_stats = {
            attr: {
                stat_name: jnp.array(v, dtype=jnp.float32)
                for stat_name, v in attr_stats.items()
            }
            for attr, attr_stats in ckpt_dict["norm_stats"].items()
        }
        self.denoiser_norm_stats = jax.tree_map(
            lambda x: jnp.expand_dims(x, 0) if len(x.shape) == 0 else x,
            self.denoiser_norm_stats,
        )
        print(f"Restored synthetic rollout generator from {args.denoiser_checkpoint}")


class MixedRolloutGenerator:
    """Rollout generator with mixed real and synthetic rollouts"""

    def __init__(
        self,
        rng,
        args,
        obs_shape,
        action_dim,
        action_lims,
        num_env_steps,
        agent_apply_fn=None,
    ):
        # TODO: remove these as input (here and others) and compute them from the dataset
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_lims = action_lims
        assert 0 < args.synth_batch_size <= args.batch_size
        self.synth_batch_size = args.synth_batch_size
        self.real_batch_size = args.batch_size - self.synth_batch_size
        self.synth_batch_lifetime = args.synth_batch_lifetime
        assert self.synth_batch_size % self.synth_batch_lifetime == 0
        self.synth_batch_size = self.synth_batch_size // self.synth_batch_lifetime
        self.synth_rollout_gens = []
        for _ in range(args.synth_batch_lifetime):
            rng, _rng = jax.random.split(rng)
            self.synth_rollout_gens.append(
                SyntheticRolloutGenerator(
                    _rng,
                    args,
                    obs_shape,
                    action_dim,
                    action_lims,
                    num_env_steps,
                    agent_apply_fn,
                    self.synth_batch_size,
                )
            )
        self.synth_batch_pointer = 0
        if self.real_batch_size > 0:
            self.real_rollout_gen = OfflineRolloutGenerator(
                args,
                obs_shape,
                action_dim,
                action_lims,
                num_env_steps,
                agent_apply_fn,
                self.real_batch_size,
            )
            self.obs_stats = self.real_rollout_gen.obs_stats
        else:
            print("WARNING: real batch size is 0, using only synthetic rollouts")

    def update_synthetic_dataset(self, rng, agent_params=None):
        # Regenerate synthetic dataset from the current agent state
        self.synth_rollout_gens[self.synth_batch_pointer].update_synthetic_dataset(
            rng, agent_params
        )
        self.synth_batch_pointer = (
            self.synth_batch_pointer + 1
        ) % self.synth_batch_lifetime

    def set_apply_fn(self, agent_apply_fn):
        for i in range(self.synth_batch_lifetime):
            self.synth_rollout_gens[i].set_apply_fn(agent_apply_fn)
        if self.real_batch_size > 0:
            self.real_rollout_gen.set_apply_fn(agent_apply_fn)

    def batch_rollout(self, rng):
        flattened_batches = []
        for i in range(self.synth_batch_lifetime):
            rng, _rng = jax.random.split(rng)
            flattened_synth_batch = self.synth_rollout_gens[i].batch_rollout(_rng)
            flattened_batches.append(flattened_synth_batch)
        if self.real_batch_size > 0:
            rng, _rng = jax.random.split(rng)
            flattened_real_batch = self.real_rollout_gen.batch_rollout(_rng)
            flattened_batches.append(flattened_real_batch)
        traj_batch = jax.tree_map(
            lambda *x: jnp.concatenate([batch for batch in x], axis=-2),
            *flattened_batches,
        )
        return jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
