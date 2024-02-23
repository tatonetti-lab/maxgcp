from typing import TypeAlias

import numpy as np  # type: ignore
import scipy.linalg  # type: ignore
from numpy.typing import ArrayLike, NDArray  # type: ignore

Array: TypeAlias = NDArray


def check_input(mat: Array):
    """Check that the input is a valid covariance matrix"""
    if mat.ndim != 2:
        raise ValueError("Input must be 2D")

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Input must be square")

    if not np.allclose(mat, mat.T, atol=1e-5):
        raise ValueError("Input must be symmetric")


def check_inputs(cov_G: Array, cov_P: Array):
    """Check that the inputs are valid covariance matrices"""
    check_input(cov_G)
    check_input(cov_P)

    if (cov_G.shape[0] != cov_G.shape[1]) or (cov_P.shape[0] != cov_P.shape[1]):
        raise ValueError("Covariance matrices must be square")


def fit_heritability(cov_G: ArrayLike, cov_P: ArrayLike) -> NDArray:
    """
    Fit a weight matrix to define phenotypes that are maximally heritable.
    Definitions are linear combinations of the phenotypes whose genetic and
    phenotypic covariance matrices are provided.

    Parameters
    ----------
    cov_G : ArrayLike of shape (n_phenotypes, n_phenotypes)
        Genetic covariance matrix.
    cov_P : ArrayLike of shape (n_phenotypes, n_phenotypes)
        Phenotypic covariance matrix.

    Returns
    -------
    weights : ndarray of shape (n_phenotypes, n_phenotypes)
        Weight matrix that defines the heritable phenotypes.
    """
    cov_G = np.asarray(cov_G)
    cov_P = np.asarray(cov_P)
    check_inputs(cov_G, cov_P)

    cov_G_sqrt: NDArray = scipy.linalg.sqrtm(cov_G)  # type: ignore
    lhs = cov_G_sqrt @ np.linalg.pinv(cov_P) @ cov_G_sqrt
    _, evecs = np.linalg.eig(lhs)
    weights = np.linalg.pinv(cov_G_sqrt) @ evecs
    weights = np.asarray(weights)

    # Normalize weights so that projections have unit variance
    weights = weights / np.sqrt(np.diag(weights.T @ cov_P @ weights))
    return weights


def fit_coheritability(cov_G: ArrayLike, cov_P: ArrayLike) -> NDArray:
    """
    Fit a weight matrix to define phenotypes that are maximally coheritable
    with the phenotypes whose genetic and phenotypic covariance matrices are
    provided.

    Parameters
    ----------
    cov_G : ArrayLike of shape (n_phenotypes, n_phenotypes)
        Genetic covariance matrix.
    cov_P : ArrayLike of shape (n_phenotypes, n_phenotypes)
        Phenotypic covariance matrix.

    Returns
    -------
    weights : ndarray of shape (n_phenotypes, n_phenotypes)
        Weight matrix that defines the coheritable phenotypes.
    """
    cov_G = np.asarray(cov_G)
    cov_P = np.asarray(cov_P)
    check_inputs(cov_G, cov_P)

    weights, _, _, _ = np.linalg.lstsq(cov_P, cov_G, rcond=None)
    weights = np.asarray(weights)

    # Normalize weights so that projections have unit variance
    weights = weights / np.sqrt(np.diag(weights.T @ cov_P @ weights))
    return weights


def fit_genetic_correlation(phenotype_idx: int, cov_G: ArrayLike) -> NDArray:
    """
    Fit a weight vector to define a phenotype that is maximally correlated
    with the phenotype at the provided index.

    Unlike the other estimators, this estimator cannot be fit for every
    phenotype, as the solution is trivially the identity matrix.

    TODO: This could actually be done by masking, not implemented.

    Parameters
    ----------
    phenotype_idx : int
        Index of the phenotype to correlate with.
    cov_G : ArrayLike of shape (n_phenotypes, n_phenotypes)
        Genetic covariance matrix.

    Returns
    -------
    weights : ndarray of shape (n_phenotypes,)
        Weight vector that defines the correlated phenotype.
    """
    cov_G = np.asarray(cov_G)
    check_input(cov_G)

    if cov_G.ndim != 2:
        raise ValueError("Covariance matrix must be 2D")

    if cov_G.shape[0] != cov_G.shape[1]:
        raise ValueError("Covariance matrix must be square")

    v = np.delete(cov_G[phenotype_idx], phenotype_idx, 0)
    G = np.delete(np.delete(cov_G, phenotype_idx, 0), phenotype_idx, 1)

    weights, _, _, _ = np.linalg.lstsq(G, v)
    return np.asarray(weights)
