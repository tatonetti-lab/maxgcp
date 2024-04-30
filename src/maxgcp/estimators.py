from typing import TypeAlias

import numpy as np  # type: ignore
import scipy.linalg  # type: ignore
from numpy.typing import NDArray  # type: ignore

Vector: TypeAlias = NDArray
Matrix: TypeAlias = NDArray


def check_matrix(mat: Matrix):
    """Check that the input is a valid covariance matrix"""
    if mat.ndim != 2:
        raise ValueError("Input must be 2D")

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Input must be square")

    if not np.allclose(mat, mat.T, atol=1e-5):
        raise ValueError("Input must be symmetric")


def check_matrix_inputs(cov_G: Matrix, cov_P: Matrix) -> None:
    """Check that the inputs are valid covariance matrices"""
    check_matrix(cov_G)
    check_matrix(cov_P)

    if (cov_G.shape[0] != cov_G.shape[1]) or (cov_P.shape[0] != cov_P.shape[1]):
        raise ValueError("Covariance matrices must be square")


def check_matrix_vector_inputs(cov_G: Vector, cov_P: Matrix) -> None:
    """Check that the inputs are a valid coheritability optimization problem"""
    check_matrix(cov_P)
    if cov_G.ndim != 1:
        raise ValueError("Input must be 1D")

    if cov_G.shape[0] != cov_P.shape[0]:
        raise ValueError("Vector must have the same length as the matrix")


def fit_heritability(cov_G: Matrix, cov_P: Matrix) -> Matrix:
    """Fit phenotypes to maximize heritability.

    Given n input phenotypes, this fits n linearly independent phenotypes that
    maximize heritability. This is akin to fitting principal components, but
    using heritability instead of overall variance as the criterion.

    Args:
        cov_G: Genetic covariance matrix (features x features)
        cov_P : Phenotypic covariance matrix (features x features)

    Returns:
        Weights that defines the phenotypes (features x features)

    Raises:
        ValueError: If the input matrices are not square or symmetric.
    """
    cov_G = np.asarray(cov_G)
    cov_P = np.asarray(cov_P)
    check_matrix_inputs(cov_G, cov_P)

    cov_G_sqrt: NDArray = scipy.linalg.sqrtm(cov_G)  # type: ignore
    lhs = cov_G_sqrt @ np.linalg.pinv(cov_P) @ cov_G_sqrt
    _, evecs = np.linalg.eig(lhs)
    weights = np.linalg.pinv(cov_G_sqrt) @ evecs
    weights = np.asarray(weights)

    # Normalize weights so that projections have unit variance
    weights = weights / np.sqrt(np.diag(weights.T @ cov_P @ weights))
    return weights


def fit_coheritability(cov_G: Vector, cov_P: Matrix) -> Matrix:
    """Fit a MaxGCP phenotype to the genetic and phenotypic covariances.

    Args:
        cov_G: Vector of genetic covariances between the target and features.
        cov_P: Phenotypic covariance matrix (features x features).

    Returns:
        A weight vector that defines the MaxGCP phenotype (features x 1)

    Raises:
        ValueError: If the input matrices are not square or symmetric.
    """
    cov_G = np.asarray(cov_G)
    cov_P = np.asarray(cov_P)
    check_matrix_vector_inputs(cov_G, cov_P)

    weights, _, _, _ = np.linalg.lstsq(cov_P, cov_G, rcond=None)
    weights = np.asarray(weights)

    # Normalize weights so that projection has unit variance
    var = weights.T @ cov_P @ weights
    weights = weights / np.sqrt(var)
    return weights


def fit_all_coheritability(cov_G: Matrix, cov_P: Matrix) -> Matrix:
    """Fit a MaxGCP phenotype to every input phenotype.

    This is equivalent to treating each phenotype as the target in a MaxGCP
    regression, and fitting the corresponding weights.

    Args:
        cov_G: Matrix of genetic covariances (features x features)
        cov_P: Matrix of phenotypic covariances (features x features)

    Returns:
        A weight matrix that define the MaxGCP phenotypes (features x features).

    Raises:
        ValueError: If the input matrices are not square or symmetric.
    """
    cov_G = np.asarray(cov_G)
    cov_P = np.asarray(cov_P)
    check_matrix_inputs(cov_G, cov_P)

    weights, _, _, _ = np.linalg.lstsq(cov_P, cov_G, rcond=None)
    weights = np.asarray(weights)

    # Normalize weights so that projections have unit variance
    weights = weights / np.sqrt(np.diag(weights.T @ cov_P @ weights))
    return weights


def fit_genetic_correlation(phenotype_idx: int, cov_G: Matrix) -> Vector:
    """Fit a phenotype that is maximally genetically correlated with the
    specified feature phenotype.

    Unlike the other methods, this does not let the target be included as a
    feature in the resulting phenotype.

    Args:
        phenotype_idx: Index of the target phenotype
        cov_G: Genetic covariance matrix (features x features)

    Returns:
        A weight vector that defines the optimized phenotype (features x 1)

    Raises:
        ValueError: If the input matrix is not square or symmetric.
    """
    cov_G = np.asarray(cov_G)
    check_matrix(cov_G)

    if cov_G.ndim != 2:
        raise ValueError("Covariance matrix must be 2D")

    if cov_G.shape[0] != cov_G.shape[1]:
        raise ValueError("Covariance matrix must be square")

    v = np.delete(cov_G[phenotype_idx], phenotype_idx, 0)
    G = np.delete(np.delete(cov_G, phenotype_idx, 0), phenotype_idx, 1)

    weights, _, _, _ = np.linalg.lstsq(G, v)

    # Normalize weights so that projection has unit variance
    var = weights.T @ G @ weights
    weights = weights / np.sqrt(var)
    return np.asarray(weights)
