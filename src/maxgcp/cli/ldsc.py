import logging
from pathlib import Path
from typing import Annotated

import ldsc.scripts.ldsc
import ldsc.scripts.munge_sumstats
import typer

logger = logging.getLogger("rich")

app = typer.Typer(
    add_completion=False, context_settings={"help_option_names": ["-h", "--help"]}
)


@app.command(name="munge")
def ldsc_munge(
    gwas_path: Annotated[
        Path,
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ],
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
) -> None:
    """Process a GWAS summary statistics file using LDSC."""
    args = ldsc.scripts.munge_sumstats.parser.parse_args(
        [
            "--sumstats",
            gwas_path.as_posix(),
            "--out",
            output_file.as_posix(),
            "--snp",
            snp_col,
            "--a1",
            a1_col,
            "--a2",
            a2_col,
            "--N-col",
            sample_size_col,
            "--p",
            p_col,
            "--signed-sumstats",
            f"{signed_sumstat_col},{signed_sumstat_null}",
        ]
    )
    ldsc.scripts.munge_sumstats.munge_sumstats(args)


@app.command(name="rg")
def ldsc_rg(
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            help="Path to munged GWAS summary statistics",
        ),
    ],
    tag_file: Annotated[
        str,
        typer.Option("--tagfile", exists=True, help="Path to tag file or directory"),
    ],
    output_stem: Annotated[
        Path,
        typer.Option("--out", help="Path to output file"),
    ],
) -> None:
    """Compute genetic covariances using LDSC."""
    args = ldsc.scripts.ldsc.parser.parse_args(
        [
            "--rg",
            ",".join(p.as_posix() for p in gwas_paths),
            "--ref-ld-chr",
            tag_file,
            "--w-ld-chr",
            tag_file,
            "--out",
            output_stem.as_posix(),
        ]
    )
    ldsc.scripts.ldsc.main(args)
