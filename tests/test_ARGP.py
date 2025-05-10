import numpy as np
from sklearn.metrics import mean_squared_error
from lazy_GP import ARGP

def test_linear_time_series():

    np.random.seed(42)  # For reproducibility
    D = 3  # Number of exogenous features
    N = 100  # Number of samples
    N_AR = 2  # Number of auto-regressive components

    # Generate random exogenous inputs
    X = np.random.rand(N, D)

    # True coefficients for exogenous inputs and AR components
    theta_exog = np.array([2.0, -1.0, 0.5]).reshape(-1, 1)  # Coefficients for exogenous inputs
    theta_ar = np.array([0.2, -0.1]).reshape(-1, 1)         # Coefficients for AR components
    true_theta = np.vstack([theta_exog, theta_ar])
    sigma = 0.001

    # Generate target values with AR components
    y = np.zeros(N)
    for t in range(N_AR, N):
        y[t] = (
            X[t] @ theta_exog +  # Contribution from exogenous inputs
            y[t-N_AR : t] @ theta_ar  # Contribution from AR components
        )

    # Observation noise
    y = y + sigma * np.random.randn(N)

    # Train using 50 points
    m = ARGP()
    theta = np.ones(5)
    X_train = X[:50, :]
    y_train = y[:50]
    m.set_hyperparameters(X=X_train, y=y_train, theta=theta, sigma=sigma, N_AR=N_AR)

    # Model predictions
    y_pred = m.predict_full_model(X[N_AR:], y0=y[:N_AR])
    mse = mean_squared_error(y[N_AR:], y_pred)
    assert mse < 0.01
