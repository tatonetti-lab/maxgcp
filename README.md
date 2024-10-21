# Maximum genetic component phenotyping

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/pypi/v/maxgcp)](https://pypi.python.org/pypi/maxgcp)
[![image](https://img.shields.io/pypi/l/maxgcp)](https://pypi.python.org/pypi/maxgcp)
[![image](https://img.shields.io/pypi/pyversions/maxgcp)](https://pypi.python.org/pypi/maxgcp)
[![Actions status](https://img.shields.io/github/actions/workflow/status/tatonetti-lab/maxgcp/test.yml?branch=main&label=actions)](https://github.com/tatonetti-lab/maxgcp/actions)

Maximum genetic component phenotyping. Optimized phenotype definitions boost GWAS power.

`maxgcp` is a Python package that implements maximum genetic component phenotyping (MaxGCP), a method that optimizes a linear phenotype definition to maximize its heritability and genetic correlation with a trait of interest.
In short, this method results in a phenotype definition that is, close to the genetic component of the trait of interest, on the individual level.
This phenotype definition can be used in various applications, including enhancement of statistical power in genome-wide association studies (GWAS).
`maxgcp` requires only estimates of genetic and phenotypic covariances, which can be obtained from GWAS summary statistics.

## Usage

```python
import maxgcp
import numpy as np

# Genetic covariances between target and feature phenotypes
genetic_cov_vec = np.array([0.5, 0.75, 0.25])
phenotypic_cov_mat = np.array([
    [0.5, 0.25, 0.1],
    [0.25, 0.3, 0.05],
    [0.1, 0.05, 0.6],
])

# Compute the MaxGCP phenotype (defined by these weights)
w = maxgcp.fit_coheritability(genetic_cov_vec, phenotypic_cov_mat)

# Evaluate MaxGCP at the individual level
feature_phenotypes = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
])
maxgcp_phenotypes = feature_phenotypes @ w

# Show the results
>>> maxgcp_phenotypes
array([-0.34242555,  0.18438299,  1.87895044,  0.        ])
```

MaxGCP fits a phenotype (y) to a target (z) to maximize coheritability between y and z.
This package can also fit a maximally heritable phenotype `maxgcp.fit_heritability` or a maximally genetically correlated phenotype (`maxgcp.fit_genetic_correlation`.

## Command-line usage

MaxGCP also exposes the `maxgcp` command in the command line.
Once `pip` installed, simply run `maxgcp --help` to see all possible commands.

Here's an example using GWAS summary statistics:

```bash
maxgcp run \
  --pcov phenotypic_covariance_matrix.csv \ # Can be computed using maxgcp pcov
  --tagfile ld_ref_panel/eur_w_ld_chr \ # LDSC tagfiles
  --target E11 \ # Which GWAS file should be the target phenotype?
  --n-covar 12 \ # How many covariates were used in the input GWAS (e.g. age+sex+PC1+...+PC10 = 12)
  --no-compress-output \ # Do not compress the output file
  --out E11.maxgcp.tsv \ # The output file to create
  E11.glm.linear I10.glm.linear gwas/*.glm.linear  # Input GWAS summary statistic files
```

To run end-to-end like this, you'll need some LDSC tagfiles.
These can be obtained from a few different places, such as [the MTAG repository](https://github.com/JonJala/mtag/tree/9e17f3cf1fbcf57b6bc466daefdc51fd0de3c5dc/ld_ref_panel).
For more information about these or the LD score regressions that will be run, see [the LDSC repository](https://github.com/bulik/ldsc), particularly [the wiki](https://github.com/bulik/ldsc/wiki).

For more information about `maxgcp run` or any other subcommand, please run `maxgcp run --help`, `maxgcp pcov --help`, etc. as appropriate.

## Installation

```bash
pip install maxgcp
```

Please see this repository's [pyproject.toml](pyproject.toml) for a full list of dependencies.
