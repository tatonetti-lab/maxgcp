import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import polars as pl
import typer
from rich.logging import RichHandler


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")


app = typer.Typer(add_completion=False)

