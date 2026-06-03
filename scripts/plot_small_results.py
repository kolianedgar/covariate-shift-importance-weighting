import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import generate_all_plots

PLOT_DIR = "results/plots/small/"

results_df = pd.read_csv(
    "results/tables/small/results.csv"
)

generate_all_plots(
    results_df,
    plot_dir=PLOT_DIR,
)

print("[DONE] Small-grid plots generated.")