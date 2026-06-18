import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# HELPER: SAVE FIGURE
# ============================================================

def save_figure(filename, plot_dir):

    path = os.path.join(plot_dir, filename)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {path}")

def check_estimator_reliability(d, lambda_vals, SAMPLE_ESTIMATOR_EXPONENT_THRESHOLD):
    """
    Warn about lambda values where d * lambda^2 exceeds the threshold,
    indicating that the sample-based chi-squared estimator is unreliable.
 
    At high d and lambda, training samples from N(0, I) have negligible
    overlap with the shifted distribution N(lambda*1, I), causing the
    Monte Carlo estimator of E[r(x)^2] to collapse toward zero even
    though the true chi-squared value exp(d * lambda^2) - 1 is large.
    """
    unreliable = [
        lam for lam in lambda_vals
        if d * lam ** 2 > SAMPLE_ESTIMATOR_EXPONENT_THRESHOLD
    ]
    if unreliable:
        true_vals = {lam: np.exp(d * lam**2) - 1 for lam in unreliable}
        print(
            f"[WARN] d={d}: sample estimator unreliable at "
            f"λ = {unreliable} (d·λ² > {SAMPLE_ESTIMATOR_EXPONENT_THRESHOLD}). "
            f"True χ² values: "
            + ", ".join(f"λ={lam} → {val:.3e}" for lam, val in true_vals.items())
            + f". With n=1000 samples from N(0,I), virtually no points land "
            f"near the shifted mean at these settings, causing the estimator "
            f"to collapse toward zero. Plotted values at these λ are unreliable."
        )

def check_non_monotone(d, subset):
    """
    Detect and warn about non-monotone behaviour in the aggregated
    chi-squared values, which indicates estimator collapse.
    Chi-squared should be monotonically increasing in lambda.
    """
    vals = subset["chi_squared_divergence_mean"].values
    lambdas = subset["lambda"].values
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            print(
                f"[WARN] d={d}: non-monotone chi-squared detected between "
                f"λ={lambdas[i-1]} (χ²={vals[i-1]:.4f}) and "
                f"λ={lambdas[i]} (χ²={vals[i]:.4f}). "
                f"This indicates sample estimator collapse, not a true decrease."
            )

def nice_x_formatter(values):
    """
    Choose a clean tick formatter based on the magnitude of the x values.
    Returns a matplotlib FuncFormatter.
    """
    max_val = values.max() if len(values) > 0 else 1.0
 
    if max_val < 0.01:
        # Scientific notation for very small values
        return ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    elif max_val < 0.1:
        return ticker.FuncFormatter(lambda x, _: f"{x:.4f}")
    elif max_val < 1.0:
        return ticker.FuncFormatter(lambda x, _: f"{x:.3f}")
    else:
        return ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    
def nice_y_formatter(values):
    """
    Choose a clean tick formatter based on the magnitude of the y values.
    Returns a matplotlib FuncFormatter.
    """
    max_val = values.max() if len(values) > 0 else 1.0
 
    if max_val < 0.01:
        return ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    elif max_val < 0.1:
        return ticker.FuncFormatter(lambda x, _: f"{x:.4f}")
    elif max_val < 1.0:
        return ticker.FuncFormatter(lambda x, _: f"{x:.3f}")
    else:
        return ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
