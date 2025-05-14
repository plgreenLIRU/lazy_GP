import numpy as np
from sklearn.metrics import mean_squared_error
from lazy_GP import GP

def generate_1D_data():

    X = np.linspace(0, 10, 50)[:, np.newaxis]
    X_star = np.linspace(0, 10, 100)[:, np.newaxis]
    
    sigma = 0.01
    y = np.sin(X[:, 0]) + sigma * np.random.randn(len(X))
    y_star = np.sin(X_star[:, 0]) + sigma * np.random.randn(len(X_star))
    return X, X_star, y, y_star

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
    X, X_star, y, y_star = generate_1D_data()

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

def test_mv_k():
    
    X, Y = generate_2D_data()[0:2]
    N = np.shape(X)[0]

    sigma = 0.01
    theta = np.array([0.1, 0.5])
    d_dash = 0

    gp = GP()

    K, C, inv_C, dK_dtheta = gp._find_exact_matrices(X, theta, sigma, d_dash)
    v = np.random.randn(N)

    assert np.allclose(C @ v, gp._mv_k(X1=X, X2=X, v=v, theta=theta, sigma=sigma, X1_equal_X2=True))    
    
def test_mv_dk():
    
    X, Y = generate_2D_data()[0:2]
    N = np.shape(X)[0]

    sigma = 0.01
    theta = np.array([1., 1.])
    d_dash = 0

    gp = GP()
    K, C, inv_C, dK_dtheta = gp._find_exact_matrices(X, theta, sigma, d_dash)
    v = np.random.randn(N)

    assert np.allclose(dK_dtheta @ v, gp._mv_dk(X, d_dash, v, theta))

def test_tr_invK_dK():
    np.random.seed(42)

    X, Y = generate_2D_data()[0:2]
    N = np.shape(X)[0]

    sigma = 0.01
    theta = np.array([2., 3.])
    d_dash = 0    

    gp = GP()
    K, C, inv_C, dK_dtheta = gp._find_exact_matrices(X=X, theta=theta, sigma=sigma, d_dash=d_dash)

    exact_term = np.trace(inv_C @ dK_dtheta)
    mc_estimate = gp._tr_invK_dK(X=X, theta=theta, sigma=sigma, d_dash=d_dash)

    assert np.allclose(exact_term, mc_estimate, rtol=0.05)

def test_dlogp():

    # Create data
    np.random.seed(42)
    X, X_star, y, y_star = generate_1D_data()

    # Initialise model
    gp = GP()

    theta = 1.5
    sigma = 0.01
    grad = gp.dlogp(X=X, y=y, theta=np.array([theta]), sigma=sigma)
    grad_exact = gp._exact_dlogp(X=X, y=y, theta=np.array([theta]), sigma=sigma)

    assert np.allclose(grad, grad_exact, rtol=0.1)
