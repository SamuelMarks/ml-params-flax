import jax.numpy as jnp
from flax import nn

from ml_params_flax.train import CNN


def create_model(key, shape=(1, 28, 28, 1), dtype=jnp.float32):
    _, initial_params = CNN.init_by_shape(key, [(shape, dtype)])
    return nn.Model(CNN, initial_params)
