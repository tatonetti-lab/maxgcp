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

## Installation

```bash
pip install maxgcp
```

`maxgcp` depends on NumPy, SciPy, and pandas.
