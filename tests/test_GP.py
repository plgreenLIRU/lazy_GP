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

def find_K_and_dK(X, theta, d_dash):
    """
    Long-winded way of evaluating K and dK / dtheta_d_dash, used for tests
    """
    N, D = np.shape(X)
    K = np.ones([N, N])
    dK_dtheta = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            dK_dtheta[i, j] = theta[d_dash]**-3
            for d in range(D):
                K[i, j] *= np.exp(-0.5 * (X[i, d] - X[j, d])**2)
                dK_dtheta[i, j] *= np.exp(-0.5 * (X[i, d] - X[j, d])**2)
            dK_dtheta[i, j] *= (X[i, d_dash] - X[j, d_dash])**2
    return K, dK_dtheta

def test_matrix_vector_products():
    
    # Generate input data
    np.random.seed(42)
    N = 100
    D = 2
    X = np.random.uniform(-3, 3, (N, D))
    sigma = 0.001
    theta = np.array([1., 1.])
    d_dash = 0

    K, dK_dtheta = find_K_and_dK(X, theta, d_dash)
    C = K + np.eye(N) * sigma**2
    inv_C = np.linalg.inv(C)
    v = np.random.randn(N)

    gp = GP()

    assert np.allclose(C @ v, gp._mv_k(X1=X, X2=X, v=v, theta=theta, sigma=0.001, X1_equal_X2=True))
    assert np.allclose(dK_dtheta @ v, gp._mv_dk(X, d_dash, v, theta))
    assert np.abs(gp._tr_invK_dK(X, theta, sigma, d_dash, S=100) - np.trace(inv_C @ dK_dtheta)) < 100
    