import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from maxgcp.loss_functions import (
    genetic_loss_mapper,
    genetic_loss_mean_mapper,
    genetic_loss_vector,
    h2_mapper,
    h2_vec,
    log_function,
    log_function_mapper,
    rg_mapper,
    rg_vec,
)


class Estimator:
    def __init__(self):
        self.n_features = None
        self.coef = None
        self._log_records = list()

    @property
    def log_df(self):
        return pd.DataFrame(self._log_records)

    def predict(self, X):
        return X @ self.coef


class SingleGeneticEstimator(Estimator):
    def _log_iteration(self, step, coef, h2_target, gamma, Cx, Px, print_iter=False):
        if print_iter:
            log_function(step, coef, gamma, Cx, Px, h2_target)

        self._log_records.append(
            {
                "step": step,
                "loss": np.asarray(genetic_loss_vector(coef, gamma, Cx, Px, h2_target)),
                "h2": np.asarray(h2_vec(coef, Cx, Px)),
                "rg": np.asarray(rg_vec(coef, gamma, Cx, h2_target)),
            }
        )

    def train(
        self,
        h2_target,
        gamma,
        Cx,
        Px,
        n_iter=1000,
        learning_rate=0.001,
        verbose=True,
        n_log=100,
    ):
        """
        Train a model to predict genetic liabilities for a single trait

        Parameters
        ----------
        h2_target : float
            Target heritability
        gamma : jax.numpy.ndarray
            Vector of genetic covariances between each feature and the target (Shape: mx1)
        Cx : jax.numpy.ndarray
            Matrix of genetic covariances between each feature (Shape: mxm)
        Px : jax.numpy.ndarray
            Matrix of phenotypic covariances between each feature (Shape: mxm)
        n_iter : int, optional
            Number of training iterations, by default 1000
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 0.001
        verbose : bool, optional
            Whether to print during training, by default True
        n_log : int, optional
            How often to log loss, heritability, and genetic correlation, by default
            100 steps
        """
        self.n_features = Cx.shape[0]
        self.coef = 1e-4 * jnp.ones((self.n_features, 1))

        genetic_loss_gradient = jax.grad(genetic_loss_vector)
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(self.coef)

        for i in range(n_iter):
            if i % n_log == 0:
                self._log_iteration(i, self.coef, h2_target, gamma, Cx, Px, verbose)

            grads = genetic_loss_gradient(self.coef, gamma, Cx, Px, h2_target)
            updates, opt_state = optimizer.update(grads, opt_state)
            self.coef = optax.apply_updates(self.coef, updates)

        # Log at the end
        self._log_iteration(i, self.coef, h2_target, gamma, Cx, Px, verbose)
        return


class AllGeneticEstimator(Estimator):
    def _log_iteration(self, step, coef_mat, gamma_mat, C, P, h2_vec, print_iter=False):
        if print_iter:
            log_function_mapper(step, coef_mat, gamma_mat, C, P, h2_vec)

        self._log_records.extend([
            {"step": step, "phenotype": p, "loss": l, "h2": h, "rg": r}
            for p, l, h, r in zip(
                range(self.n_features),
                np.asarray(genetic_loss_mapper(coef_mat, gamma_mat, C, P, h2_vec)),
                np.asarray(h2_mapper(coef_mat, C, P)),
                np.asarray(rg_mapper(coef_mat, gamma_mat, C, h2_vec)),
            )
        ])

    def train(self, C, P, n_iter=1000, learning_rate=0.001, verbose=True, n_log=100):
        """
        Train a model to predict genetic liabilities for all traits

        Parameters
        ----------
        C : jax.numpy.ndarray
            Matrix of genetic covariances (Shape: mxm)
        P : jax.numpy.ndarray
            Matrix of phenotypic covariances (Shape: mxm)
        n_iter : int, optional
            Number of training iterations, by default 1000
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 0.001
        verbose : bool, optional
            Whether to print during training, by default True
        n_log : int, optional
            How often to log loss, heritability, and genetic correlation, by default
            100 steps
        """
        self.n_features = C.shape[0]

        # Create normalizer to zero coefficients on the diagonal
        normalizer = 1 - jnp.eye(self.n_features)
        self.coef = 1e-4 * normalizer

        # Create matrix and vector needed for training.
        # Gamma is the matrix of feature-to-target genetic covariance vectors
        # h2v is the vector of feature heritabilities
        gam = normalizer * C
        h2v = jnp.diag(C)

        genetic_loss_gradient = jax.grad(genetic_loss_mean_mapper)
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(self.coef)

        for i in range(n_iter):
            if i % n_log == 0:
                self._log_iteration(i, self.coef, gam, C, P, h2v, verbose)

            grads = genetic_loss_gradient(self.coef, gam, C, P, h2v)
            updates, opt_state = optimizer.update(grads * normalizer, opt_state)
            self.coef = optax.apply_updates(self.coef, updates)

        # Log at the end
        self._log_iteration(i, self.coef, gam, C, P, h2v, verbose)
        return
