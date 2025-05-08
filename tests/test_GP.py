import numpy as np
from sklearn.metrics import mean_squared_error
from lazy_GP import GP

def test_1D_regression():

    # Create data
    np.random.seed(42)
    sigma = 0.05
    X = np.linspace(0, 10, 50)[:, np.newaxis]
    X_star = np.linspace(0, 10, 100)[:, np.newaxis]
    y = np.sin(X[:, 0]) + sigma * np.random.randn(len(X))
    y_star = np.sin(X_star[:, 0]) + sigma * np.random.randn(len(X_star))

    # Fit model
    m = GP()
    theta = np.array([1])
    m.set_hyperparameters(X=X, y=y, theta=theta, sigma=0.1)
    y_star_pred = m.predict(X_star)

    # Check mse
    mse = mean_squared_error(y_star, y_star_pred)
    assert mse < 0.005

def test_2D_regression():

    # Define a 2D nonlinear function
    def nonlinear_function(x):
        return np.sin(x[:, 0]) * np.cos(x[:, 1])

    # Generate input data
    np.random.seed(42)
    N = 500
    X = np.random.uniform(-3, 3, (N, 2))
    X_star = np.random.uniform(-3, 3, (100, 2))

    # Generate noisy observations
    sigma = 0.001
    y = nonlinear_function(X) + sigma * np.random.randn(len(X))
    y_star = nonlinear_function(X_star) + sigma * np.random.randn(len(X_star))

    # Initialise model
    m = GP()
    theta = np.array([1., 1.])
    m.set_hyperparameters(X, y, theta=theta, sigma=sigma)

    # Predictions
    y_star_pred = m.predict(X_star)
    mse = mean_squared_error(y_star, y_star_pred)
    assert mse < 0.002
