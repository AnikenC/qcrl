import timeit
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit, config
from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    LinearInterpolation,
    SaveAt,
    PIDController,
)

config.update("jax_enable_x64", True)

# TO-DO
# Complete this simulation according to the Overleaf notes
