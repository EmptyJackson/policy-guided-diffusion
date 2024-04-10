import distrax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


def normalize(x, stats):
    return (x - stats["mean"]) / (stats["std"] + 1e-3)


class SoftQNetwork(nn.Module):
    activation: str = "relu"
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
        return q


class TanhDeterministicActor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"
    action_lims: Sequence[float] = (-1.0, 1.0)
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
        action = nn.Dense(self.action_dim)(x)
        action_scale = (self.action_lims[1] - self.action_lims[0]) / 2
        action_bias = (self.action_lims[1] + self.action_lims[0]) / 2
        pi = distrax.Transformed(
            distrax.Deterministic(action),
            # Note: Chained bijectors applied in reverse order
            distrax.Chain(
                [
                    distrax.ScalarAffine(action_bias, action_scale),
                    distrax.Tanh(),
                ]
            ),
        )
        return pi
