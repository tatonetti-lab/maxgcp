import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pymbend.rayleigh
import typer

from maxgcp.estimators import fit_heritability

logger = logging.getLogger("rich")

app = typer.Typer(
    add_completion=False, context_settings={"help_option_names": ["-h", "--help"]}
)


@app.command(name="maxh")
def compare_maxh(
    phenotype_covariance_file: Annotated[
        Path,
        typer.Option("--pcov", exists=True, help="Path to phenotypic covariance file"),
    ],
    genetic_covariance_file: Annotated[
        Path,
        typer.Option("--gcov", exists=True, help="Path to genetic covariance file"),
    ],
    output_file: Annotated[
        Path, typer.Option("--out", help="Path to output GWAS summary statistics file")
    ],
):
    """Run MaxH on a set of GWAS summary statistics."""
    logger.info("Loading phenotypic covariance matrix")
    sep = "," if phenotype_covariance_file.suffix == ".csv" else "\t"
    phenotypic_covariance_df = pd.read_csv(
        phenotype_covariance_file, sep=sep, index_col=0
    )
    if (
        phenotypic_covariance_df.index.values.tolist()
        != phenotypic_covariance_df.columns.values.tolist()
    ):
        raise ValueError("Phenotypic covariance matrix must be symmetric")
    logger.info("Loading genetic covariance matrix")
    genetic_covariance_df = pd.read_csv(genetic_covariance_file, sep="\t", index_col=0)
    if (
        genetic_covariance_df.index.values.tolist()
        != genetic_covariance_df.columns.values.tolist()
    ):
        raise ValueError("Genetic covariance matrix must be symmetric")
    if genetic_covariance_df.shape[0] != phenotypic_covariance_df.shape[0]:
        raise ValueError(
            "Genetic and phenotypic covariance matrices must have the same "
            "number of features"
        )
    if set(genetic_covariance_df.columns.tolist()) != set(
        phenotypic_covariance_df.index.tolist()
    ):
        raise ValueError(
            "Genetic and phenotypic covariance matrices must have the same features"
        )
    # Sort the features identically
    phenotypes = phenotypic_covariance_df.index.tolist()
    genetic_covariance_df = genetic_covariance_df.loc[phenotypes, phenotypes]
    phenotypic_covariance_df = phenotypic_covariance_df.loc[phenotypes, phenotypes]
    if not all(np.linalg.eigvals(genetic_covariance_df.values) > 0):
        logger.info("Genetic covariance matrix is not positive definite. Bending...")
        genetic_covariance = pymbend.rayleigh.bend_generalized_rayleigh(
            genetic_covariance_df.values, phenotypic_covariance_df.values
        )
        genetic_covariance_df = pd.DataFrame(
            genetic_covariance,
            index=genetic_covariance_df.index,
            columns=genetic_covariance_df.columns,
        )
    logger.info("Fitting MaxH")
    try:
        maxh_weights = fit_heritability(
            genetic_covariance_df.values, phenotypic_covariance_df.values
        )
    except ValueError as e:
        raise ValueError(f"Could not fit MaxH: {e}") from e
    logger.info("Saving weights")
    index = pd.Index(phenotypes, name="phenotype")
    columns = pd.Index(
        [f"MaxH_{i}" for i in range(1, len(phenotypes) + 1)], name="projection"
    )
    maxh_weights_df = pd.DataFrame(maxh_weights, index=index, columns=columns)
    maxh_weights_df.to_csv(output_file, sep="\t")
    logger.info("Done")
