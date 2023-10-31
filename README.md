# Maximum genetic component phenotyping

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`maxgcp` is a Python package that implements maximum genetic component phenotyping (MaxGCP), a method that optimizes a linear phenotype definition to maximize its heritability and genetic correlation with a trait of interest.
In short, this method results in a phenotype definition that is, close to the genetic component of the trait of interest, on the individual level.
This phenotype definition can be used in various applications, including enhancement of statistical power in genome-wide association studies (GWAS).
`maxgcp` requires only estimates of genetic and phenotypic covariances, which can be obtained from GWAS summary statistics.
