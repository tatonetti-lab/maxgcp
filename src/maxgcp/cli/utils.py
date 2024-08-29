import concurrent.futures
import logging
from pathlib import Path

from rich.progress import track

from maxgcp.cli.ldsc import ldsc_munge, ldsc_rg

logger = logging.getLogger("rich")


def remove_all_suffixes(path: Path) -> Path:
    while path.suffixes:
        path = path.with_suffix("")
    return path


def run_munge(args: tuple[Path, Path, str, str, str, str, str, str, float]) -> Path:
    (
        gwas_path,
        output_dir,
        snp_col,
        a1_col,
        a2_col,
        sample_size_col,
        p_col,
        signed_sumstat_col,
        signed_sumstat_null,
    ) = args
    output_root = output_dir.joinpath(gwas_path.name)
    output_path = output_dir.joinpath(gwas_path.name + ".sumstats.gz")
    ldsc_munge(
        gwas_path,
        output_root,
        snp_col=snp_col,
        a1_col=a1_col,
        a2_col=a2_col,
        sample_size_col=sample_size_col,
        p_col=p_col,
        signed_sumstat_col=signed_sumstat_col,
        signed_sumstat_null=signed_sumstat_null,
    )
    return output_path


def munge_parallel(
    gwas_paths: list[Path],
    output_dir: Path,
    snp_col: str,
    a1_col: str,
    a2_col: str,
    sample_size_col: str,
    p_col: str,
    signed_sumstat_col: str,
    signed_sumstat_null: float,
    n_threads: int,
) -> list[Path]:
    args = [
        (
            gwas_path,
            output_dir,
            snp_col,
            a1_col,
            a2_col,
            sample_size_col,
            p_col,
            signed_sumstat_col,
            signed_sumstat_null,
        )
        for gwas_path in gwas_paths
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
        output_paths = list(
            track(
                executor.map(run_munge, args),
                total=len(args),
                description="Formatting sumstats...",
            )
        )
    return output_paths


def run_rg(args: tuple[Path, list[Path], str, Path]) -> Path:
    (
        target,
        gwas_paths,
        tag_file,
        directory,
    ) = args
    sorted_paths = [target] + [p for p in gwas_paths if p != target]
    output_stem = directory.joinpath(remove_all_suffixes(target).stem)
    ldsc_rg(
        gwas_paths=sorted_paths,
        tag_file=tag_file,
        output_stem=output_stem,
    )
    output_path = output_stem.with_suffix(".log")
    if not output_path.exists():
        raise ValueError(f"RG output file {output_path} not found")
    return output_path


def rg_parallel(
    gwas_paths: list[Path],
    targets: list[Path],
    tag_file: str,
    directory: Path,
    n_threads: int,
) -> list[Path]:
    args = [(target, gwas_paths, tag_file, directory) for target in targets]
    if n_threads == 1:
        return list(
            track(
                (run_rg(a) for a in args),
                description="Computing RG...",
                total=len(args),
            )
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
        output_paths = list(
            track(
                executor.map(run_rg, args),
                total=len(args),
                description="Computing genetic covariances...",
            )
        )
    return output_paths
