from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def train_linear_regression(
    X_train,
    y_train
):
    """
    Train Ordinary Least Squares (OLS) regression.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted OLS model
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model

def train_weighted_linear_regression(
    X_train,
    y_train,
    weights
):
    """
    Train importance-weighted linear regression.

    Minimises:

        sum_i w_i (y_i - x_i^T beta)^2

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    weights : np.ndarray
        Importance weights of shape (n_samples,)

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted weighted linear regression model
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return model

def train_linear_svr(
    X_train,
    y_train,
    C=1.0,
    epsilon=0.1
):
    """
    Train Linear Support Vector Regression.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    Returns
    -------
    model : sklearn.svm.SVR
        Trained linear SVR model
    """

    model = SVR(
        kernel="linear",
        C=C,
        epsilon=epsilon
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def train_weighted_linear_svr(
    X_train,
    y_train,
    weights,
    C=1.0,
    epsilon=0.1,
    max_iter=-1
):
    """
    Train importance-weighted Linear SVR.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    weights : np.ndarray
        Importance weights of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    max_iter : int
        Maximum optimisation iterations

    Returns
    -------
    model : sklearn.svm.SVR
        Trained weighted Linear SVR model
    """

    model = SVR(
        kernel="linear",
        C=C,
        epsilon=epsilon,
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return model

def train_rbf_svr(
    X_train,
    y_train,
    C=1.0,
    epsilon=0.1,
    gamma="scale",
    max_iter=-1
):
    """
    Train Gaussian RBF-kernel Support Vector Regression.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    gamma : str or float
        RBF kernel coefficient

    max_iter : int
        Maximum optimisation iterations

    Returns
    -------
    model : sklearn.svm.SVR
        Trained RBF-SVR model
    """

    model = SVR(
        kernel="rbf",
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def train_weighted_rbf_svr(
    X_train,
    y_train,
    weights,
    C=1.0,
    epsilon=0.1,
    gamma="scale",
    max_iter=-1
):
    """
    Train importance-weighted Gaussian RBF-kernel SVR.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d)

    y_train : np.ndarray
        Training targets of shape (n_samples,)

    weights : np.ndarray
        Importance weights of shape (n_samples,)

    C : float
        Regularisation parameter

    epsilon : float
        Epsilon-insensitive tube width

    gamma : str or float
        RBF kernel bandwidth parameter

    max_iter : int
        Maximum optimisation iterations

    Returns
    -------
    model : sklearn.svm.SVR
        Trained weighted RBF-SVR model
    """

    model = SVR(
        kernel="rbf",
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return model

def predict_model(model, X):
    """
    Generate predictions using a fitted model.

    Parameters
    ----------
    model :
        Trained sklearn model.

    X : np.ndarray
        Input features of shape (n_samples, d)

    Returns
    -------
    y_pred : np.ndarray
        Predicted targets of shape (n_samples,)
    """

    y_pred = model.predict(X)

    return y_pred
