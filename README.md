# Maximum genetic component phenotyping

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Actions status](https://img.shields.io/github/actions/workflow/status/tatonetti-lab/maxgcp/test.yml?branch=main&label=actions)](https://github.com/tatonetti-lab/maxgcp/actions)

`maxgcp` is a Python package that implements maximum genetic component phenotyping (MaxGCP), a method that optimizes a linear phenotype definition to maximize its heritability and genetic correlation with a trait of interest.
In short, this method results in a phenotype definition that is, close to the genetic component of the trait of interest, on the individual level.
This phenotype definition can be used in various applications, including enhancement of statistical power in genome-wide association studies (GWAS).
`maxgcp` requires only estimates of genetic and phenotypic covariances, which can be obtained from GWAS summary statistics.
