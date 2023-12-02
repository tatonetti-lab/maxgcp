import functools

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
import optax  # type: ignore
import pandas as pd  # type: ignore

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
    def __init__(self, cov_G_X, cov_P_X):
        self.cov_G_X = cov_G_X
        self.cov_P_X = cov_P_X
        self.n_features = cov_G_X.shape[0]
        self.coef = None
        self._log_records = list()
        self.opt_state = None

    @property
    def log_df(self):
        return pd.DataFrame(self._log_records)

    @property
    def coefficients(self) -> np.ndarray:
        if self.coef is None:
            raise ValueError("Model has not been trained yet")

        return np.asarray(self.coef)

    def predict(self, X):
        return X @ self.coef


class SingleGeneticEstimator(Estimator):
    def __init__(self, cov_G_X, cov_P_X):
        super().__init__(cov_G_X, cov_P_X)

        # Initialize coefficients to 1e-4
        self.coef = 1e-4 * jnp.ones((self.n_features, 1))

        # Set up loss function
        self.loss_fn = functools.partial(
            genetic_loss_vector,
            cov_G_X=cov_G_X,
            cov_P_X=cov_P_X,
        )

        # Gradient of loss function
        self.loss_grad = jax.grad(self.loss_fn)

    def _log_iteration(self, step, coef, h2_target, gamma, print_iter=False):
        if print_iter:
            log_function(step, coef, gamma, self.cov_G_X, self.cov_P_X, h2_target)

        self._log_records.append(
            {
                "step": step,
                "loss": np.asarray(
                    genetic_loss_vector(
                        coef, gamma, self.cov_G_X, self.cov_P_X, h2_target
                    )
                ),
                "h2": np.asarray(h2_vec(coef, self.cov_G_X, self.cov_P_X)),
                "rg": np.asarray(rg_vec(coef, gamma, self.cov_G_X, h2_target)),
            }
        )

    def train(
        self,
        var_G_z,
        cov_G_z_X,
        n_iter=1000,
        learning_rate=0.001,
        reset_state=True,
        verbose=True,
        n_log=100,
    ):
        """
        Train a model to predict genetic liabilities for a single trait

        Parameters
        ----------
        var_G_z : float
            Genetic variance of the target phenotype
        cov_G_z_X : jax.numpy.ndarray
            Vector of genetic covariances between features and the target (Shape: mx1)
        n_iter : int, optional
            Number of training iterations, by default 1000
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 0.001
        reset_state : bool, optional
            Whether to reset the optimizer state, by default True. If False, and the
            model has been trained before, the previous optimizer state will be used to
            continue training. Care should be taken when setting this to False, as the
            optimizer state may not apply to the current target phenotype.
        verbose : bool, optional
            Whether to print during training, by default True
        n_log : int, optional
            How often to log loss, heritability, and genetic correlation, by default
            100 steps
        """
        optimizer = optax.adam(learning_rate=learning_rate)

        if reset_state or self.opt_state is None:
            self.opt_state = optimizer.init(self.coef)

        for i in range(n_iter):
            if i % n_log == 0:
                self._log_iteration(i, self.coef, var_G_z, cov_G_z_X, verbose)

            grads = self.loss_grad(self.coef, cov_G_z_X, var_G_z)
            updates, self.opt_state = optimizer.update(grads, self.opt_state)
            self.coef = optax.apply_updates(self.coef, updates)

        # Log at the end
        self._log_iteration(n_iter, self.coef, var_G_z, cov_G_z_X, verbose)
        return


class AllGeneticEstimator(Estimator):
    def __init__(self, cov_G_X, cov_P_X):
        super().__init__(cov_G_X, cov_P_X)

        # Create normalizer to zero coefficients on the diagonal
        self.normalizer = 1 - jnp.eye(self.n_features)

        # Initialize coefficients to 1e-4, but zero out the diagonal
        self.coef = 1e-4 * self.normalizer

        # Genetic covariance between features and targets (simply ignores self terms)
        self.cov_G_Z_X = self.normalizer * cov_G_X

        # Set up loss function
        self.loss_fn = functools.partial(
            genetic_loss_mean_mapper,
            cov_G_Z_X=self.cov_G_Z_X,
            cov_G_X=cov_G_X,
            cov_P_X=cov_P_X,
        )

        # Gradient of loss function
        self.loss_grad = jax.grad(self.loss_fn)

    def _log_iteration(self, step, beta_mat, print_iter=False):
        if print_iter:
            log_function_mapper(
                step, beta_mat, self.cov_G_Z_X, self.cov_G_X, self.cov_P_X
            )

        self._log_records.extend(
            [
                {"step": step, "phenotype": p, "loss": loss, "h2": h, "rg": r}
                for p, loss, h, r in zip(
                    range(self.n_features),
                    np.asarray(
                        genetic_loss_mapper(
                            beta_mat, self.cov_G_Z_X, self.cov_G_X, self.cov_P_X
                        )
                    ),
                    np.asarray(h2_mapper(beta_mat, self.cov_G_X, self.cov_P_X)),
                    np.asarray(rg_mapper(beta_mat, self.cov_G_Z_X, self.cov_G_X)),
                )
            ]
        )

    def train(
        self,
        n_iter=1000,
        learning_rate=1e-5,
        reset_state=False,
        verbose=True,
        n_log=100,
    ):
        """
        Train a model to infer the genetic component of each phenotype

        Parameters
        ----------
        n_iter : int, optional
            Number of training iterations, by default 1000
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 0.001
        reset_state : bool, optional
            Whether to reset the optimizer state, by default False. If False, and the
            model has been trained before, the previous optimizer state will be used to
            continue training.
        verbose : bool, optional
            Whether to print during training, by default True
        n_log : int, optional
            How often to log loss, heritability, and genetic correlation, by default
            100 steps
        """
        optimizer = optax.adam(learning_rate=learning_rate)

        if reset_state or self.opt_state is None:
            self.opt_state = optimizer.init(self.coef)

        for i in range(n_iter):
            if i % n_log == 0:
                self._log_iteration(i, self.coef, verbose)

            # Compute and normalize gradients
            grads = self.loss_grad(self.coef) * self.normalizer
            updates, self.opt_state = optimizer.update(grads, self.opt_state)
            self.coef = optax.apply_updates(self.coef, updates)

        # Log at the end
        self._log_iteration(n_iter, self.coef, verbose)
        return
