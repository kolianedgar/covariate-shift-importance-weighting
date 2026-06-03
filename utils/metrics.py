import numpy as np

def mse(y_true, y_pred):
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets

    y_pred : np.ndarray
        Predicted targets

    Returns
    -------
    float
        Mean Squared Error
    """

    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets

    y_pred : np.ndarray
        Predicted targets

    Returns
    -------
    float
        Root Mean Squared Error
    """

    return np.sqrt(mse(y_true, y_pred))
