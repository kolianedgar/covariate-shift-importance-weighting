import os
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# HELPER: SAVE FIGURE
# ============================================================

def save_figure(filename, plot_dir):

    path = os.path.join(plot_dir, filename)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {path}")