import numpy as np
from sklearn.metrics import mean_squared_error
from lazy_GP import GP


def generate_2D_data():

    # Define a 2D nonlinear function
    def nonlinear_function(x):
        return np.sin(x[:, 0]) * np.cos(x[:, 1])

    # Generate input data
    N = 100
    X = np.random.uniform(-3, 3, (N, 2))
    X_star = np.random.uniform(-3, 3, (100, 2))

    # Generate noisy observations
    sigma = 0.01
    y = nonlinear_function(X) + sigma * np.random.randn(len(X))
    y_star = nonlinear_function(X_star) + sigma * np.random.randn(len(X_star))
    return X, X_star, y, y_star

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

    np.random.seed(42)
    X, X_star, y, y_star = generate_2D_data()

    # Initialise model
    m = GP()
    theta = np.array([1., 1.])
    sigma = 0.01
    m.set_hyperparameters(X, y, theta=theta, sigma=sigma)

    # Predictions
    y_star_pred = m.predict(X_star)
    mse = mean_squared_error(y_star, y_star_pred)
    assert mse < 0.005

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

def test_mv_k():
    
    X, Y = generate_2D_data()[0:2]
    N = np.shape(X)[0]

    sigma = 0.01
    theta = np.array([1., 1.])
    d_dash = 0

    K, dK_dtheta = find_K_and_dK(X, theta, d_dash)
    C = K + np.eye(N) * sigma**2
    inv_C = np.linalg.inv(C)
    v = np.random.randn(N)

    gp = GP()

    assert np.allclose(C @ v, gp._mv_k(X1=X, X2=X, v=v, theta=theta, sigma=sigma, X1_equal_X2=True))    
    
def test_mv_dk():
    
    X, Y = generate_2D_data()[0:2]
    N = np.shape(X)[0]

    sigma = 0.01
    theta = np.array([1., 1.])
    d_dash = 0

    K, dK_dtheta = find_K_and_dK(X, theta, d_dash)
    C = K + np.eye(N) * sigma**2
    inv_C = np.linalg.inv(C)
    v = np.random.randn(N)

    gp = GP()

    assert np.allclose(dK_dtheta @ v, gp._mv_dk(X, d_dash, v, theta))

def test_tr_invK_dK():
    np.random.seed(42)

    X, Y = generate_2D_data()[0:2]
    N = np.shape(X)[0]

    sigma = 0.01
    theta = np.array([1., 1.])
    d_dash = 0    

    K, dK_dtheta = find_K_and_dK(X, theta, d_dash)
    C = K + np.eye(N) * sigma**2
    inv_C = np.linalg.inv(C)

    gp = GP()

    exact_term = np.trace(inv_C @ dK_dtheta)
    mc_estimate = gp._tr_invK_dK(X=X, theta=theta, sigma=sigma, d_dash=d_dash, S=50)

    assert np.allclose(exact_term, mc_estimate, atol=20)
