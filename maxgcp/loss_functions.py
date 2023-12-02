import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import Array, jit


@jit
def genetic_loss_vector(
    beta: Array, cov_G_z_X: Array, cov_G_X: Array, cov_P_X: Array, var_G_z: Array
) -> Array:
    h2_term = (beta.T @ cov_G_X @ beta) / (beta.T @ cov_P_X @ beta)
    rg_term = (beta.T @ cov_G_z_X) / jnp.sqrt(var_G_z * beta.T @ cov_G_X @ beta)
    return jnp.abs(1 - rg_term) + jnp.abs(1 - h2_term)


@jit
def genetic_loss_mean_mapper(coefficient_matrix, cov_G_Z_X, cov_G_X, cov_P_X) -> Array:
    return jax.vmap(genetic_loss_vector, (1, 1, None, None, 0))(
        coefficient_matrix, cov_G_Z_X, cov_G_X, cov_P_X, jnp.diag(cov_G_X)
    ).mean()


def genetic_loss_mapper(coefficient_matrix, cov_G_Z_X, cov_G_X, cov_P_X) -> Array:
    return jax.vmap(genetic_loss_vector, (1, 1, None, None, 0))(
        coefficient_matrix, cov_G_Z_X, cov_G_X, cov_P_X, jnp.diag(cov_G_X)
    )


def h2_vec(beta, cov_G_X, cov_P_X):
    # Var_G(y) = sum_{i,j} beta_i beta_j Cov(g_i, g_j)
    # Var_P(y) = sum_{i,j} beta_i beta_j Cov(x_i, x_j)
    # h2(y) = Var_G(y) / Var_P(y)
    return (beta.T @ cov_G_X @ beta) / (beta.T @ cov_P_X @ beta)


def rg_vec(beta, cov_G_z_X, cov_G_X, g_var_z):
    # Var_G(y) = sum_{i,j} beta_i beta_j Cov(g_i, g_j)
    # Var_G(z) = g_var_z
    # Cov_G(z, y) = sum_i beta_i Cov(g_i, g_z)
    # rg(z, y) = Cov_G(z, y) / sqrt(Var_G(z) * Var_G(y))
    return (beta.T @ cov_G_z_X) / jnp.sqrt((beta.T @ cov_G_X @ beta) * g_var_z)


def h2_mapper(beta_matrix, cov_G_X, cov_P_X):
    return jax.vmap(h2_vec, (1, None, None))(beta_matrix, cov_G_X, cov_P_X)


def rg_mapper(beta_matrix, cov_G_Z_X, cov_G_X):
    return jax.vmap(rg_vec, (1, 1, None, 0))(
        beta_matrix, cov_G_Z_X, cov_G_X, jnp.diag(cov_G_X)
    )


def log_function(step, beta, cov_G_z_X, cov_G_X, cov_P_X, g_var_z):
    genetic_loss = genetic_loss_vector(beta, cov_G_z_X, cov_G_X, cov_P_X, g_var_z)
    h2 = h2_vec(beta, cov_G_X, cov_P_X)
    rg = rg_vec(beta, cov_G_z_X, cov_G_X, g_var_z)
    print(
        f"Step: {step:<5}\tMean loss: {genetic_loss:.5f}\tMean h2: {h2:.4f}\tMean rg: "
        f"{rg:.4f}"
    )


def log_function_mapper(step, beta_matrix, cov_G_Z_X, cov_G_X, cov_P_X):
    genetic_loss_mean = genetic_loss_mean_mapper(
        beta_matrix, cov_G_Z_X, cov_G_X, cov_P_X
    ).item()
    h2_mean = h2_mapper(beta_matrix, cov_G_X, cov_P_X).mean().item()
    rg_mean = rg_mapper(beta_matrix, cov_G_Z_X, cov_G_X).mean().item()
    print(
        f"Step: {step:<5}\tMean loss: {genetic_loss_mean:.5f}\tMean h2: {h2_mean:.4f}\t"
        f"Mean rg: {rg_mean:.4f}"
    )
