import jax
import jax.numpy as jnp
from flax import linen as nn

# we are going to use ResNet with preNorm.
# see https://arxiv.org/abs/1603.05027 for details
class ResidualBlock(nn.Module):
    num_filters: int

    @nn.compact
    def __call__(self, x, is_training: bool):
        residual = x
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = nn.relu(x)
        x = nn.Conv(self.num_filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = nn.relu(x)
        x = nn.Conv(self.num_filters, kernel_size=(3, 3), padding='SAME')(x)
        return x + residual

class AZNet(nn.Module):
    """AlphaZero-like network for Chess with policy and value heads."""
    num_actions: int
    num_filters: int = 128
    num_residual_blocks: int = 8

    @nn.compact
    def __call__(self, x, is_training: bool):
        x = x.astype(jnp.float32)
        x = nn.Conv(self.num_filters, kernel_size=(3, 3), padding='SAME')(x)

        for _ in range(self.num_residual_blocks):
            x = ResidualBlock(self.num_filters)(x, is_training)

        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = nn.relu(x)

        # Policy head
        p = nn.Conv(2, kernel_size=(1, 1), padding='SAME')(x)
        p = nn.BatchNorm(use_running_average=not is_training)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))
        p_logits = nn.Dense(self.num_actions)(p)

        # Value head
        v = nn.Conv(1, kernel_size=(1, 1), padding='SAME')(x)
        v = nn.BatchNorm(use_running_average=not is_training)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1)) # flatten
        v = nn.Dense(self.num_filters)(v)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return p_logits, v
