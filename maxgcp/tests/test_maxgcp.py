from dataclasses import dataclass

import numpy as np  # type: ignore
import pytest

import maxgcp.estimators


@dataclass
class RawData:
    n_samples: int
    n_traits: int
    n_variants: int
    genotypes: np.ndarray
    phenotypes: np.ndarray
    effect_sizes: np.ndarray
    genetic_covariance: np.ndarray
    phenotypic_covariance: np.ndarray


def simulate_raw_data(
    n_samples: int, n_traits: int, n_variants: int, seed: int
) -> RawData:
    np.random.seed(seed)

    genotypes = np.random.randint(0, 3, size=(n_samples, n_variants))
    effect_sizes = np.random.normal(size=(n_variants, n_traits))
    genetic_liability = genotypes @ effect_sizes

    environmental_noise = np.random.normal(size=(n_samples, n_traits))
    phenotypes = genetic_liability + environmental_noise

    genetic_covariance = np.cov(genetic_liability, rowvar=False)
    phenotypic_covariance = np.cov(phenotypes, rowvar=False)

    return RawData(
        n_samples=n_samples,
        n_traits=n_traits,
        n_variants=n_variants,
        genotypes=genotypes,
        phenotypes=phenotypes,
        effect_sizes=effect_sizes,
        genetic_covariance=genetic_covariance,
        phenotypic_covariance=phenotypic_covariance,
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_maxgcp_loss_form(
    seed, n_traits=100, n_iter=1000, learning_rate=1e-5, verbose=True
):
    """
    Evaluate whether flattening the loss function terms changes the optimized
    coefficients.
    """
    # These don't change anything about MaxGCP
    n_samples = 100
    n_variants = 100

    raw_data = simulate_raw_data(n_samples, n_traits, n_variants, seed)
    G = raw_data.genetic_covariance
    P = raw_data.phenotypic_covariance

    stacked_estimator = maxgcp.estimators.AllGeneticEstimator(G, P, flatten_loss=False)
    stacked_estimator.train(n_iter=n_iter, learning_rate=learning_rate, verbose=verbose)

    flat_estimator = maxgcp.estimators.AllGeneticEstimator(G, P, flatten_loss=True)
    flat_estimator.train(n_iter=n_iter, learning_rate=learning_rate, verbose=verbose)

    max_difference = np.abs(
        stacked_estimator.coefficients - flat_estimator.coefficients
    ).max()
    assert max_difference == pytest.approx(0.0, abs=1e-4)  # FAILS!


@pytest.mark.parametrize("seed", [0, 1])
def test_gradients_identical(seed):
    """
    Evaluate whether flattening the loss function terms changes the gradients.

    Since proportional gradients are irrelevant to the optimization, we
    check whether the gradients are identical up to a constant factor.
    """
    # These don't change anything about MaxGCP
    n_samples = 100
    n_traits = 100
    n_variants = 100

    raw_data = simulate_raw_data(n_samples, n_traits, n_variants, seed)
    G = raw_data.genetic_covariance
    P = raw_data.phenotypic_covariance

    stacked_estimator = maxgcp.estimators.AllGeneticEstimator(G, P, flatten_loss=False)
    flat_estimator = maxgcp.estimators.AllGeneticEstimator(G, P, flatten_loss=True)

    np.random.seed(seed)
    random_coef_1 = np.random.normal(size=(n_traits, n_traits))
    random_coef_2 = np.random.normal(size=(n_traits, n_traits))

    stacked_grad = stacked_estimator.loss_grad(random_coef_1)
    flat_grad = flat_estimator.loss_grad(random_coef_1)
    factor_1 = stacked_grad / flat_grad

    stacked_grad = stacked_estimator.loss_grad(random_coef_2)
    flat_grad = flat_estimator.loss_grad(random_coef_2)
    factor_2 = stacked_grad / flat_grad

    max_difference = np.abs(factor_1 - factor_2).max()
    assert max_difference == pytest.approx(0.0, abs=1e-4)  # FAILS!
