import logging
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import polars as pl
import typer
from igwas.igwas import igwas_files
from rich.logging import RichHandler

from maxgcp.cli.compare import app as compare_app
from maxgcp.cli.ldsc import app as ldsc_app
from maxgcp.cli.ldsc import ldsc_rg
from maxgcp.cli.utils import munge_parallel, remove_all_suffixes, rg_parallel
from maxgcp.estimators import fit_coheritability

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")


app = typer.Typer(
    add_completion=False, context_settings={"help_option_names": ["-h", "--help"]}
)
app.add_typer(ldsc_app, name="ldsc", help="LDSC commands")
app.add_typer(compare_app, name="compare", help="Compare methods")


@app.command(name="pcov")
def compute_phenotypic_covariance(
    phenotype_file: Annotated[
        Path,
        typer.Option(
            "--pheno", exists=True, help="Path to phenotype file", show_default=False
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ],
    covariate_file: Annotated[
        Optional[Path],
        typer.Option("--covar", exists=True, help="Path to covariate file"),
    ] = None,
    person_id_col: Annotated[
        list[str], typer.Option("--person-id", help="Person ID column(s)")
    ] = ["#FID", "IID"],
    no_intercept: Annotated[
        bool, typer.Option("--no-intercept", help="Do not add intercept to covariates")
    ] = False,
) -> None:
    """Compute phenotypic covariance matrix from a phenotype file.

    Covariates are optional. When provided, the covariates will be residualized
    out and the resulting file will contain the partial covariance matrix.

    Person ID columns will be used to join the phenotype and covariate files and
    will be ignored in the computation of the covariance matrix. Multiple ID
    can be specified one or more times like this: --person-id FID --person-id IID
    """
    add_intercept = not no_intercept
    logger.info("Computing covariance")
    if covariate_file is None and add_intercept:
        logger.warning(
            "You have specified to add an intercept to the covariates, but no "
            "covariate file was provided. No intercept will be added."
        )
    sep = "," if phenotype_file.suffix == ".csv" else "\t"
    phenotype_df = pl.read_csv(phenotype_file, separator=sep)
    logger.info(
        f"Found {phenotype_df.shape[0]} samples, {phenotype_df.shape[1]} phenotypes"
    )
    has_person_ids = person_id_col is not None and len(person_id_col) > 0
    if has_person_ids:
        phenotype_names = phenotype_df.drop(person_id_col).columns
    else:
        phenotype_names = phenotype_df.columns
    if covariate_file is not None:
        logger.info("Residualizing covariates")
        sep = "," if covariate_file.suffix == ".csv" else "\t"
        covariate_df = pl.read_csv(covariate_file, separator="\t")
        if add_intercept:
            covariate_df = covariate_df.with_columns(
                pl.lit(1.0).alias("const").cast(pl.Float64)
            )
        has_person_ids = person_id_col is not None and len(person_id_col) > 0
        if has_person_ids:
            covariate_names = covariate_df.drop(person_id_col).columns
        else:
            covariate_names = covariate_df.columns
        merged_df = phenotype_df.join(covariate_df, on=person_id_col)
        X = merged_df.select(covariate_names).to_numpy()
        Y = merged_df.select(phenotype_names).to_numpy()
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Y_resid = Y - X @ beta
        covariance = np.cov(Y_resid.T)
    else:
        X = phenotype_df.select(phenotype_names).to_numpy()
        covariance = np.cov(X.T)

    index = pd.Index(phenotype_names, name="phenotype")
    covariance_df = pd.DataFrame(covariance, index=index, columns=index)
    logger.info(f"Writing covariance to {output_file}")
    covariance_df.to_csv(output_file, sep="\t")


@app.command(name="pcov-vec")
def compute_phenotypic_variances(
    phenotype_file: Annotated[
        Path,
        typer.Option(
            "--pheno", exists=True, help="Path to phenotype file", show_default=False
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ],
    covariate_file: Annotated[
        Optional[Path],
        typer.Option("--covar", exists=True, help="Path to covariate file"),
    ] = None,
    person_id_col: Annotated[
        list[str], typer.Option("--person-id", help="Person ID column(s)")
    ] = ["#FID", "IID"],
    no_intercept: Annotated[
        bool, typer.Option("--no-intercept", help="Do not add intercept to covariates")
    ] = False,
) -> None:
    """Compute phenotypic variances from a phenotype file.

    Covariates are optional. When provided, the covariates will be residualized
    out and the resulting file will contain the partial variances.

    Person ID columns will be used to join the phenotype and covariate files and
    will be ignored in the computation of the variances. Multiple ID can be
    specified one or more times like this: --person-id FID --person-id IID
    """
    add_intercept = not no_intercept
    logger.info("Computing covariance")
    if covariate_file is None and add_intercept:
        logger.warning(
            "You have specified to add an intercept to the covariates, but no "
            "covariate file was provided. No intercept will be added."
        )
    sep = "," if phenotype_file.suffix == ".csv" else "\t"
    phenotype_df = pl.read_csv(phenotype_file, separator=sep)
    logger.info(
        f"Found {phenotype_df.shape[0]} samples, {phenotype_df.shape[1]} phenotypes"
    )
    has_person_ids = person_id_col is not None and len(person_id_col) > 0
    if has_person_ids:
        phenotype_names = phenotype_df.drop(person_id_col).columns
    else:
        phenotype_names = phenotype_df.columns
    if covariate_file is not None:
        logger.info("Residualizing covariates")
        sep = "," if covariate_file.suffix == ".csv" else "\t"
        covariate_df = pl.read_csv(covariate_file, separator="\t")
        if add_intercept:
            covariate_df = covariate_df.with_columns(
                pl.lit(1.0).alias("const").cast(pl.Float64)
            )
        has_person_ids = person_id_col is not None and len(person_id_col) > 0
        if has_person_ids:
            covariate_names = covariate_df.drop(person_id_col).columns
        else:
            covariate_names = covariate_df.columns
        merged_df = phenotype_df.join(covariate_df, on=person_id_col)
        X = merged_df.select(covariate_names).to_numpy()
        Y = merged_df.select(phenotype_names).to_numpy()
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Y_resid = Y - X @ beta
        variance = np.var(Y_resid, axis=0)
    else:
        X = phenotype_df.select(phenotype_names).to_numpy()
        variance = np.var(X, axis=0)

    index = pd.Index(phenotype_names)
    variance_df = pd.Series(variance.astype(np.float32), index=index).to_frame()
    logger.info(f"Writing variance to {output_file}")
    variance_df.to_csv(output_file, sep="\t", index=True, header=False)


def read_ldsc_gcov_output(
    output_path: Path, target_phenotypic_variance: float
) -> pd.DataFrame:
    """Read the genetic covariance estimates from the LDSC output file."""
    with open(output_path) as f:
        lines = f.readlines()

    files = None
    heritabilities = list()
    genetic_covariances = list()
    state = None
    for line in lines:
        if state is None:
            if line.startswith("--rg "):
                files = line.replace("--rg ", "").replace(" \\", "").split(",")
            elif line.startswith("Heritability of phenotype "):
                state_str = line.replace("Heritability of phenotype ", "").split("/")[0]
                state = int(state_str)
            continue
        if line.startswith("Total Observed scale h2: "):
            h2_str = line.replace("Total Observed scale h2: ", "").split()[0]
            h2 = float(h2_str)
            heritabilities.append(h2)
            assert len(heritabilities) == state, f"{len(heritabilities)} != {state}"
            if state == 1:
                genetic_covariances.append(h2 * target_phenotypic_variance)
                state = None
        elif line.startswith("Total Observed scale gencov: "):
            gcov_str = line.replace("Total Observed scale gencov: ", "").split()[0]
            genetic_covariances.append(float(gcov_str))
            assert (
                len(genetic_covariances) == state
            ), f"{len(genetic_covariances)} != {state}"
            state = None

    if files is None:
        raise ValueError("Could not find files in LDSC output")

    # Remove file suffixes that were added during this process
    files = [Path(f).with_suffix("").with_suffix("").name for f in files]
    return (
        pd.DataFrame(
            {
                "target": [files[0]] * len(heritabilities),
                "feature": files,
                "heritability": heritabilities,
                "genetic_covariance": genetic_covariances,
            }
        )
        .pivot(index="feature", columns="target", values="genetic_covariance")
        .rename_axis(columns=None)
    )


@app.command(name="gcov-matrix")
def compute_genetic_covariance(
    *,
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    targets: Annotated[
        Optional[list[Path]],
        typer.Option(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ] = None,
    target_phenotypic_variance_file: Annotated[
        Path,
        typer.Option(
            "--target-pvar",
            exists=True,
            help="Path to target phenotypic variance file",
            show_default=False,
        ),
    ],
    tag_file: Annotated[
        Path,
        typer.Option("--tagfile", exists=True, help="Path to tag file"),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ] = Path("maxgcp_genetic_covariance.tsv"),
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    a1_col: Annotated[
        str, typer.Option("--a1", help="Name of effect allele column")
    ] = "A1",
    a2_col: Annotated[
        str, typer.Option("--a2", help="Name of non-effect allele column")
    ] = "OMITTED",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    p_col: Annotated[str, typer.Option("--p", help="Name of p-value column")] = "P",
    signed_sumstat_col: Annotated[
        str,
        typer.Option(
            "--signed-sumstat", help="Name of signed sumstat column (e.g. Z, OR)"
        ),
    ] = "T_STAT",
    signed_sumstat_null: Annotated[
        float,
        typer.Option(
            "--signed-sumstat-null", help="Null value for the signed sumstat column"
        ),
    ] = 0.0,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads for LDSC")
    ] = 1,
) -> None:
    """Compute a genetic covariance matrix (features x targets) using LDSC."""
    sep = "," if target_phenotypic_variance_file.suffix == ".csv" else "\t"
    name_to_pvar = (
        pd.read_csv(target_phenotypic_variance_file, sep=sep, index_col=0, header=None)
        .iloc[:, 0]
        .to_dict()
    )
    targets = gwas_paths if targets is None else targets

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        logger.info("Formatting sumstats for LDSC")
        munged_paths = munge_parallel(
            gwas_paths=gwas_paths,
            output_dir=tmpdir,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            sample_size_col=sample_size_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            n_threads=n_threads,
        )
        munged_targets = [
            p
            for p in munged_paths
            if remove_all_suffixes(p).stem
            in {remove_all_suffixes(t).stem for t in targets}
        ]
        if len(munged_targets) != len(targets):
            raise ValueError(
                f"Munged files do not match targets: {len(munged_targets)} != "
                f"{len(targets)}"
            )

        logger.info(f"Got tag file: {tag_file}, is dir: {tag_file.is_dir()}")
        tag_path = (
            tag_file.as_posix() + "/" if tag_file.is_dir() else tag_file.as_posix()
        )
        logger.info("Computing genetic covariances using LDSC")
        rg_directory = tmpdir.joinpath("rg")
        rg_directory.mkdir()
        rg_log_paths = rg_parallel(
            gwas_paths=munged_paths,
            targets=munged_targets,
            tag_file=tag_path,
            directory=rg_directory,
            n_threads=n_threads,
        )
        logger.info("Extracting genetic covariances")
        full_rg_df = None
        for log_path in rg_log_paths:
            target_phenotypic_variance = name_to_pvar[log_path.stem]
            result_df = read_ldsc_gcov_output(log_path, target_phenotypic_variance)
            if use_stem:
                result_df.index = pd.Index(
                    [remove_all_suffixes(Path(p)) for p in result_df.index],
                    name="phenotype",
                )
                result_df.columns = [
                    remove_all_suffixes(Path(p)) for p in result_df.columns
                ]
            full_rg_df = result_df if full_rg_df is None else full_rg_df.join(result_df)

        assert full_rg_df is not None, "No results found"
        full_rg_df.to_csv(output_file, sep="\t")


@app.command(name="gcov-vec")
def compute_genetic_covariance_vector(
    *,
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    target: Annotated[
        Path, typer.Option(exists=True, help="Target phenotype(s) for MaxGCP")
    ],
    target_phenotypic_variance: Annotated[
        float,
        typer.Option(
            "--target-phenotypic-variance",
            help="Variance of the target phenotype",
        ),
    ],
    tag_file: Annotated[
        Path,
        typer.Option("--tagfile", exists=True, help="Path to tag file"),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ] = Path("maxgcp_genetic_covariance.tsv"),
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    a1_col: Annotated[
        str, typer.Option("--a1", help="Name of effect allele column")
    ] = "A1",
    a2_col: Annotated[
        str, typer.Option("--a2", help="Name of non-effect allele column")
    ] = "OMITTED",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    p_col: Annotated[str, typer.Option("--p", help="Name of p-value column")] = "P",
    signed_sumstat_col: Annotated[
        str,
        typer.Option(
            "--signed-sumstat", help="Name of signed sumstat column (e.g. Z, OR)"
        ),
    ] = "T_STAT",
    signed_sumstat_null: Annotated[
        float,
        typer.Option(
            "--signed-sumstat-null", help="Null value for the signed sumstat column"
        ),
    ] = 0.0,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads for LDSC")
    ] = 1,
) -> None:
    """Compute a genetic covariance vector (features x target) using LDSC."""
    if target not in gwas_paths:
        raise ValueError(f"Target {target} not found in GWAS paths")

    gwas_paths = [target] + [p for p in gwas_paths if p != target]

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        logger.info("Formatting sumstats for LDSC")
        fmt_dir = tmpdir.joinpath("formatted")
        fmt_dir.mkdir()
        output_paths = munge_parallel(
            gwas_paths=gwas_paths,
            output_dir=fmt_dir,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            sample_size_col=sample_size_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            n_threads=n_threads,
        )
        logger.info("Computing genetic covariances using LDSC")
        logger.info(f"Got tag file: {tag_file}, is dir: {tag_file.is_dir()}")
        tag_path = (
            tag_file.as_posix() + "/" if tag_file.is_dir() else tag_file.as_posix()
        )
        temp_output_path = tmpdir.joinpath("ldsc_output.log")
        ldsc_rg(
            output_paths,
            tag_path,
            temp_output_path.with_suffix(""),
        )
        # Format the results into a table
        result_df = read_ldsc_gcov_output(temp_output_path, target_phenotypic_variance)

    if use_stem:
        result_df.index = pd.Index(
            [remove_all_suffixes(Path(p)) for p in result_df.index], name="phenotype"
        )
        result_df.columns = [remove_all_suffixes(Path(p)) for p in result_df.columns]

    result_df.to_csv(output_file, sep="\t")


@app.command(name="fit-to-gcov")
def fit_command(
    genetic_covariance_file: Annotated[
        Path,
        typer.Option("--gcov", exists=True, help="Path to genetic covariance file"),
    ],
    phenotypic_covariance_file: Annotated[
        Path,
        typer.Option("--pcov", exists=True, help="Path to phenotypic covariance file"),
    ],
    target: Annotated[str, typer.Option(help="Target phenotype for MaxGCP")],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ] = Path("maxgcp_weights.tsv"),
    include_target: Annotated[
        bool, typer.Option(help="Include target phenotype in fit")
    ] = True,
):
    """Fit MaxGCP using existing genetic and phenotypic covariances."""
    logger.info("Fitting MaxGCP phenotype")
    sep = "," if genetic_covariance_file.suffix == ".csv" else "\t"
    genetic_covariance_df = pd.read_csv(genetic_covariance_file, sep=sep, index_col=0)
    sep = "," if phenotypic_covariance_file.suffix == ".csv" else "\t"
    phenotypic_covariance_df = pd.read_csv(
        phenotypic_covariance_file, sep=sep, index_col=0
    )
    if (
        phenotypic_covariance_df.index.values.tolist()
        != phenotypic_covariance_df.columns.values.tolist()
    ):
        raise ValueError("Phenotypic covariance matrix must be symmetric")
    if target not in genetic_covariance_df.columns:
        raise ValueError(f"Target {target} not found in genetic covariance file")
    if include_target and (
        target not in phenotypic_covariance_df.columns
        or target not in phenotypic_covariance_df.index
    ):
        raise ValueError(f"Target {target} not found in phenotypic covariance file")

    if include_target:
        features = phenotypic_covariance_df.index.tolist()
    else:
        original_features = phenotypic_covariance_df.index
        features = original_features.drop(target).tolist()

    gcov_vec = genetic_covariance_df.loc[features, target].values
    pcov_mat = phenotypic_covariance_df.loc[features, features].values
    logger.info(f"Using {len(features)} features")
    maxgcp_weights = fit_coheritability(gcov_vec, pcov_mat)
    maxgcp_weights_df = pd.DataFrame(
        maxgcp_weights,
        index=pd.Index(features, name="feature"),
        columns=pd.Index([target], name="target"),
    )
    logger.info(f"Writing weights to {output_file}")
    maxgcp_weights_df.to_csv(output_file, sep="\t")


@app.command(name="igwas")
def run_indirect_gwas(
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    projection_coefficient_file: Annotated[
        Path,
        typer.Option("--proj", exists=True, help="Path to projection coefficient file"),
    ],
    phenotype_covariance_file: Annotated[
        Path,
        typer.Option("--pcov", exists=True, help="Path to phenotypic covariance file"),
    ],
    n_covar: Annotated[
        int,
        typer.Option("--n-covar", help="Number of covariates to use"),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file"),
    ],
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    beta_col: Annotated[
        str, typer.Option("--beta", help="Name of beta column")
    ] = "BETA",
    std_error_col: Annotated[
        str, typer.Option("--std-error", help="Name of standard error column")
    ] = "SE",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    compress: Annotated[
        bool, typer.Option("--compress", help="Compress output file")
    ] = True,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    chunksize: Annotated[
        int, typer.Option("--chunksize", help="Chunksize for IGWAS")
    ] = 100_000,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads for IGWAS")
    ] = 1,
):
    """Compute GWAS summary statistics for a projected phenotype."""
    if not use_stem:
        raise NotImplementedError(
            "Indirect GWAS only currently supports GWAS files where the file "
            "stem represents the phenotype"
        )
    projections = pd.read_csv(
        projection_coefficient_file, sep="\t", index_col=0, nrows=0
    ).columns.tolist()
    write_projection = len(projections) > 1
    igwas_files(
        projection_matrix_path=projection_coefficient_file.as_posix(),
        covariance_matrix_path=phenotype_covariance_file.as_posix(),
        gwas_result_paths=[p.as_posix() for p in gwas_paths],
        output_file_path=output_file.as_posix(),
        num_covar=n_covar,
        chunksize=chunksize,
        variant_id=snp_col,
        beta=beta_col,
        std_error=std_error_col,
        sample_size=sample_size_col,
        num_threads=n_threads,
        capacity=n_threads,
        compress=compress,
        quiet=True,
        write_phenotype_id=write_projection,
    )


@app.command(name="run")
def run_command(
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    phenotype_covariance_file: Annotated[
        Path,
        typer.Option("--pcov", exists=True, help="Path to phenotypic covariance file"),
    ],
    tag_file: Annotated[
        Path,
        typer.Option("--tagfile", exists=True, help="Path to tag file"),
    ],
    output_file: Annotated[
        Path, typer.Option("--out", help="Path to output GWAS summary statistics file")
    ],
    target: Annotated[
        Path, typer.Option(exists=True, help="Target phenotype for MaxGCP")
    ],
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    beta_col: Annotated[
        str, typer.Option("--beta", help="Name of beta column")
    ] = "BETA",
    std_error_col: Annotated[
        str, typer.Option("--std-error", help="Name of standard error column")
    ] = "SE",
    a1_col: Annotated[
        str, typer.Option("--a1", help="Name of effect allele column")
    ] = "A1",
    a2_col: Annotated[
        str, typer.Option("--a2", help="Name of non-effect allele column")
    ] = "OMITTED",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    p_col: Annotated[str, typer.Option("--p", help="Name of p-value column")] = "P",
    signed_sumstat_col: Annotated[
        str,
        typer.Option(
            "--signed-sumstat", help="Name of signed sumstat column (e.g. Z, OR)"
        ),
    ] = "T_STAT",
    signed_sumstat_null: Annotated[
        float,
        typer.Option(
            "--signed-sumstat-null", help="Null value for the signed sumstat column"
        ),
    ] = 0.0,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    n_covar: Annotated[
        int, typer.Option("--n-covar", help="Number of covariates to use")
    ] = 0,
    compress_output: Annotated[bool, typer.Option(help="Compress output file")] = True,
    chunksize: Annotated[
        int, typer.Option("--chunksize", help="Chunksize for IGWAS")
    ] = 100_000,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads for IGWAS")
    ] = 1,
    include_target: Annotated[
        bool, typer.Option(help="Include target phenotype in fit")
    ] = True,
    clean_up: Annotated[bool, typer.Option(help="Clean up intermediate files")] = True,
):
    """Run MaxGCP on a set of GWAS summary statistics."""
    logger.info("Computing genetic covariances using LDSC")
    sep = "," if phenotype_covariance_file.suffix == ".csv" else "\t"
    phenotypic_covariance_df = pd.read_csv(
        phenotype_covariance_file, sep=sep, index_col=0
    )
    target_name = remove_all_suffixes(target).stem
    target_phenotypic_variance = phenotypic_covariance_df.loc[
        target_name, target_name
    ].item()
    with (
        tempfile.NamedTemporaryFile(suffix=".tsv") as covariance_file,
        tempfile.NamedTemporaryFile(suffix=".tsv") as maxgcp_weights_file,
    ):
        covariance_path = Path(covariance_file.name)
        compute_genetic_covariance_vector(
            gwas_paths=gwas_paths,
            target=target,
            target_phenotypic_variance=target_phenotypic_variance,
            tag_file=tag_file,
            output_file=covariance_path,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            sample_size_col=sample_size_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            use_stem=use_stem,
        )
        maxgcp_weights_path = Path(maxgcp_weights_file.name)
        fit_command(
            genetic_covariance_file=covariance_path,
            phenotypic_covariance_file=phenotype_covariance_file,
            target=target_name,
            output_file=maxgcp_weights_path,
            include_target=include_target,
        )
        logger.info("Computing GWAS summary statistics for the MaxGCP phenotype")
        run_indirect_gwas(
            gwas_paths=gwas_paths,
            projection_coefficient_file=maxgcp_weights_path,
            phenotype_covariance_file=phenotype_covariance_file,
            n_covar=n_covar,
            output_file=output_file,
            snp_col=snp_col,
            beta_col=beta_col,
            std_error_col=std_error_col,
            sample_size_col=sample_size_col,
            compress=compress_output,
            use_stem=use_stem,
            chunksize=chunksize,
            n_threads=n_threads,
        )
        if not clean_up:
            logger.info(
                "Keeping intermediate files ['maxgcp_genetic_covariance.tsv', "
                "'maxgcp_weights.tsv']"
            )
            covariance_path.rename(
                output_file.parent.joinpath("maxgcp_genetic_covariance.tsv")
            )
            maxgcp_weights_path.rename(
                output_file.parent.joinpath("maxgcp_weights.tsv")
            )
            # Create empty files to avoid errors when cleaning up
            covariance_path.touch()
            maxgcp_weights_path.touch()
        else:
            logger.info("Cleaning up intermediate files")
    logger.info("Done")
