import jax
import jax.numpy as jnp
from jax import jit


@jit
def genetic_loss_vector(beta, gamma, C, P, h2_target):
    target_g_var = beta.T @ C @ beta
    target_p_var = beta.T @ P @ beta
    h2 = target_g_var / target_p_var
    rg = (beta.T @ gamma) / jnp.sqrt(h2_target * target_g_var)
    return jnp.abs(1 - rg) + jnp.abs(1 - h2)


@jit
def genetic_loss_mean_mapper(coefficient_matrix, gamma_matrix, C, P, h2_vector):
    return jax.vmap(genetic_loss_vector, (1, 1, None, None, 0))(
        coefficient_matrix, gamma_matrix, C, P, h2_vector
    ).mean()


def genetic_loss_mapper(coefficient_matrix, gamma_matrix, C, P, h2_vector):
    return jax.vmap(genetic_loss_vector, (1, 1, None, None, 0))(
        coefficient_matrix, gamma_matrix, C, P, h2_vector
    )


def h2_vec(beta, C, P):
    target_g_var = beta.T @ C @ beta
    target_p_var = beta.T @ P @ beta
    h2 = target_g_var / target_p_var
    return h2


def rg_vec(beta, gamma, C, h2_target):
    target_g_var = beta.T @ C @ beta
    g_cov = beta.T @ gamma
    rg = g_cov / jnp.sqrt(h2_target * target_g_var)
    return rg


def h2_mapper(coefficient_matrix, C, P):
    return jax.vmap(h2_vec, (1, None, None))(coefficient_matrix, C, P)


def rg_mapper(coefficient_matrix, gamma_matrix, C, h2_vector):
    return jax.vmap(rg_vec, (1, 1, None, 0))(
        coefficient_matrix, gamma_matrix, C, h2_vector
    )


def log_function(step, coefficient_vec, gamma_vec, Cx, Px, h2_target):
    genetic_loss = genetic_loss_vector(coefficient_vec, gamma_vec, Cx, Px, h2_target)
    h2 = h2_vec(coefficient_vec, Cx, Px)
    rg = rg_vec(coefficient_vec, gamma_vec, Cx, h2_target)
    print(
        f"Step: {step:<5}\tMean loss: {genetic_loss:.5f}\tMean h2: {h2:.4f}\tMean rg: "
        f"{rg:.4f}"
    )


def log_function_mapper(step, coefficient_matrix, gamma_matrix, C, P, h2_vector):
    genetic_loss_mean = genetic_loss_mean_mapper(
        coefficient_matrix, gamma_matrix, C, P, h2_vector
    ).item()
    h2_mean = h2_mapper(coefficient_matrix, C, P).mean().item()
    rg_mean = rg_mapper(coefficient_matrix, gamma_matrix, C, h2_vector).mean().item()
    print(
        f"Step: {step:<5}\tMean loss: {genetic_loss_mean:.5f}\tMean h2: {h2_mean:.4f}\t"
        f"Mean rg: {rg_mean:.4f}"
    )
