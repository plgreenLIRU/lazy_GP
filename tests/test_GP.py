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
    m.set_hyperparameters(X=X, y=y, theta=1, sigma=0.1)
    y_star_pred = m.predict(X_star)

    # Check mse
    mse = mean_squared_error(y_star, y_star_pred)
    assert mse < 0.005
