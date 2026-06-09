"""Redraw only the two circle figures from existing CSV matrices (no recomputation)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from local_axes_experiments.exp13_decomposition_validity.exp13_decomposition_validity import (
    EMOTIONS_ORDERED,
    save_plutchik_circle_figure,
    save_mds_circle_figure,
)

RES = Path(__file__).parent / "results"
FIG = ROOT / "thesis" / "figures"

mat_b = pd.read_csv(RES / "cosine_matrix_before_L13.csv", index_col=0).values
mat_a = pd.read_csv(RES / "cosine_matrix_after_L13.csv",  index_col=0).values

print("Redrawing caa_decomposition_plutchik_circle_L13.pdf ...")
save_plutchik_circle_figure(mat_b, mat_a, FIG / "caa_decomposition_plutchik_circle_L13.pdf")

print("Redrawing caa_decomposition_mds_circle_L13.pdf ...")
save_mds_circle_figure(mat_b, mat_a, FIG / "caa_decomposition_mds_circle_L13.pdf")

print("Done.")
