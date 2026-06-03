import torch

def generate_targets(
    X,
    beta,
    sigma,
    mode="linear"
):
    """
    Generate regression targets.

    Model:
        Y = f(X) + tau

    where:
        tau ~ N(0, sigma^2)

    Parameters
    ----------
    X : torch.Tensor
        Input data of shape (n_samples, d)

    beta : torch.Tensor
        Coefficient vector of shape (d,)

    sigma : float
        Noise standard deviation

    mode : str
        One of:
            - "linear"
            - "nonlinear"

    Returns
    -------
    y : torch.Tensor
        Target vector of shape (n_samples,)
    """

    # ============================================================
    # 1. COMPUTE SIGNAL
    # ============================================================

    linear_response = X @ beta

    if mode == "linear":

        signal = linear_response

    elif mode == "nonlinear":

        # smooth nonlinear target
        signal = torch.sin(linear_response)

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    # ============================================================
    # 2. ADD GAUSSIAN NOISE
    # ============================================================

    noise = sigma * torch.randn(
        X.shape[0],
        dtype=X.dtype,
        device=X.device
    )

    y = signal + noise

    return y
