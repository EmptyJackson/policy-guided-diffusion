import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def normalize(x, stats):
    return (x - stats["mean"]) / (stats["std"] + 1e-3)


class SoftQNetwork(nn.Module):
    activation: str = "tanh"
    obs_stats: dict = None

    @nn.compact
    def __call__(self, obs, action):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if self.obs_stats:
            obs = normalize(obs, self.obs_stats)
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.Dense(256)(x)
        x = activation(x)
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, axis=-1)


class VectorCritic(nn.Module):
    activation: str = "tanh"
    n_critics: int = 2
    obs_stats: dict = None

    @nn.compact
    def __call__(self, obs, action):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(activation=self.activation, obs_stats=self.obs_stats)(
            obs, action
        )
        return q_values


class ValueFunction(nn.Module):
    activation: str = "tanh"
    obs_stats: dict = None
    NO_ACTION_INPUT: None = None

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if self.obs_stats:
            x = normalize(x, self.obs_stats)
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


class TanhGaussianActor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    action_lims: Sequence[float] = (-1.0, 1.0)
    obs_stats: dict = None
    eval: bool = False

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if self.obs_stats:
            x = normalize(x, self.obs_stats)
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        action_scale = (self.action_lims[1] - self.action_lims[0]) / 2
        action_bias = (self.action_lims[1] + self.action_lims[0]) / 2
        mean = action_bias + action_scale * x
        logstd = self.param(
            "logstd",
            init_fn=lambda key: jnp.zeros(self.action_dim, dtype=jnp.float32),
        )
        std = jnp.exp(jnp.clip(logstd, LOG_STD_MIN, LOG_STD_MAX))
        pi = distrax.Deterministic(mean) if self.eval else distrax.Normal(mean, std)
        return pi
