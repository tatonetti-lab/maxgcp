import numpy as np
import pandas as pd


def conditional_heritability(weights, G, P):
    return np.diag(weights.T @ G @ weights) / np.diag(weights.T @ P @ weights)


def conditional_rg(weights, G):
    return (G * weights).sum(axis=0) / np.sqrt(
        np.diag(G) * np.diag(weights.T @ G @ weights)
    )


def conditional_coheritability(weights, G, P):
    return (G * weights).sum(axis=0) / np.sqrt(
        np.diag(P) * np.diag(weights.T @ P @ weights)
    )


def summary_metrics(weights, G, P):
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
