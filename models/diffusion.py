import jax
import jax.numpy as jnp
from flax import linen as nn

from util import *


class TimeEmbedding(nn.Module):
    features: int = 64

    @nn.compact
    def __call__(self, t):
        # Transformer sinusoidal positional encoding
        half_dim = self.features // 8
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.outer(t, emb)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        emb = nn.Dense(self.features)(emb)
        # Swish activation
        emb = emb * nn.sigmoid(emb)
        emb = nn.Dense(self.features)(emb)
        return emb


class Encoder(nn.Module):
    features: int = 64
    n_blocks: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, x, t):
        zs = []
        t = t * nn.sigmoid(t)
        for i in range(self.n_blocks):
            x = nn.Conv(self.features * (2**i), kernel_size=(3,))(x)
            t_emb = nn.Dense(self.features * (2**i))(t)
            x += t_emb
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            x = nn.Conv(self.features * (2**i), kernel_size=(3,))(x)
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            zs.append(x)
            x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        return zs


class Decoder(nn.Module):
    out_features: int
    features: int = 64
    n_blocks: int = 5
    n_groups: int = 8

    def _upsample(self, x, target_length):
        # Deconvolution currently just duplicates elements
        # TODO: Test alternative upsampling methods
        return jax.image.resize(
            x, shape=(*x.shape[:-2], target_length, x.shape[-1]), method="nearest"
        )

    @nn.compact
    def __call__(self, zs, t):
        x = zs[-1]
        t = t * nn.sigmoid(t)
        for i in range(self.n_blocks - 2, -1, -1):
            z = zs[i]
            x = self._upsample(x, z.shape[-2])
            x = nn.Conv(self.features * (2**i), kernel_size=(2,))(x)
            x += nn.Dense(self.features * (2**i))(t)
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            x = jnp.concatenate([x, z], axis=-1)
            x = nn.Conv(self.features * (2**i), kernel_size=(3,))(x)
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            x = nn.Conv(self.features * (2**i), kernel_size=(3,))(x)
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
        x = nn.Conv(self.out_features, kernel_size=(1,))(x)
        return x


class UNet(nn.Module):
    features: int = 64
    n_blocks: int = 5

    @nn.compact
    def __call__(self, x, t):
        t = TimeEmbedding(features=self.features)(t)
        zs = Encoder(features=self.features, n_blocks=self.n_blocks)(x, t)
        y = Decoder(
            out_features=x.shape[-1], features=self.features, n_blocks=self.n_blocks
        )(zs, t)
        return y
