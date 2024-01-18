import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from numpy.typing import NDArray  # type: ignore


def conditional_heritability(weights: NDArray, G: NDArray, P: NDArray) -> NDArray:
    return np.diag(weights.T @ G @ weights) / np.diag(weights.T @ P @ weights)


def conditional_rg(weights: NDArray, G: NDArray) -> NDArray:
    return (G * weights).sum(axis=0) / np.sqrt(
        np.diag(G) * np.diag(weights.T @ G @ weights)
    )


def conditional_coheritability(weights: NDArray, G: NDArray, P: NDArray) -> NDArray:
    return (G * weights).sum(axis=0) / np.sqrt(
        np.diag(P) * np.diag(weights.T @ P @ weights)
    )


def summary_metrics(weights: NDArray, G: NDArray, P: NDArray) -> pd.DataFrame:
    wtPw = np.diag(weights.T @ P @ weights)
    wtGw = np.diag(weights.T @ G @ weights)
    oGw = (G * weights).sum(axis=0)

    h2 = wtGw / wtPw
    rg = oGw / np.sqrt(np.diag(G) * wtGw)
    coher = oGw / np.sqrt(np.diag(P) * wtPw)
    loss = np.abs(h2 - 1) + np.abs(rg - 1)
    return pd.DataFrame(
        {
            "phenotype_idx": np.arange(len(h2)),
            "h2": h2,
            "rg": rg,
            "coher": coher,
            "loss": loss,
        }
    )
