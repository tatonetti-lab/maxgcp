import jax.numpy as jnp  # type: ignore

from maxgcp.loss_functions import h2_vec

# import pytest


def test_h2_vec():
    # Complete independence
    beta = jnp.array([1, 1])
    cov_G_X = jnp.array([[1, 0], [0, 1]])
    cov_P_X = jnp.array([[1, 0], [0, 1]])
    assert h2_vec(beta, cov_G_X, cov_P_X) == 1.0

    # Complete dependence
    beta = jnp.array([1, 1])
    cov_G_X = jnp.array([[1, 1], [1, 1]])
    cov_P_X = jnp.array([[1, 1], [1, 1]])
    assert h2_vec(beta, cov_G_X, cov_P_X) == 1.0

    # Partial dependence
    """
    x1 = g1 + e1
    x2 = g2 + e2

    var(x1) = var(g1 + e1)
            = var(g1) + var(e1)
            = 1 + 1
            = 2

    cov(x1, x2) = cov(g1 + e1, g2 + e2)
                = cov(g1, g2) + cov(g1, e2) + cov(e1, g2) + cov(e1, e2)
                = cov(g1, g2) + 0 + 0 + 0
                = cov(g1, g2)
                = 0.5

    y = x1 + x2

    var(y) = var(x1 + x2)
           = var(x1) + var(x2) + 2 * cov(x1, x2)
           = 2 + 2 + 2 * 0.5
           = 5

    var(gy) = var(g1 + g2)
            = var(g1) + var(g2) + 2 * cov(g1, g2)
            = 1 + 1 + 2 * 0.5
            = 3

    h2 = var(gy) / var(y)
       = 3 / 5
       = 0.6
    """
    beta = jnp.array([1, 1])
    cov_G_X = jnp.array([[1, 0.5], [0.5, 1]])
    cov_P_X = jnp.array([[2, 0.5], [0.5, 2]])
    assert h2_vec(beta, cov_G_X, cov_P_X) == 0.6
