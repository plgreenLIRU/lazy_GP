import numpy as np

class GP():
    """
    A class implementing a lazy Gaussian Process (GP) with RBF ARD kernel.
    """

    def _mv_k(self, X1, X2, v, theta, sigma, X1_equal_X2):
        """
        Lazily computes K(X1, X2; theta) @ v for RBF ARD kernel without forming K.

        Parameters:
        - X1: np.ndarray, shape (N1, D)
            First set of input points.
        - X2: np.ndarray, shape (N2, D)
            Second set of input points.
        - v: np.ndarray, shape (N2,)
            Vector to multiply with the kernel matrix.
        - theta: float
            Length-scales for the RBF kernel.
        - sigma: float
            Noise standard deviation
        - X1_equal_X2: bool
            Whether X1 and X2 are the same (used for adding noise to the diagonal).

        Returns:
        - result: np.ndarray, shape (N1,)
            Result of the matrix-vector multiplication.
        """
        X1 = np.vstack(X1)
        X2 = np.vstack(X2)

        N, D = X1.shape
        result = np.zeros(N)
        for n in range(N):
            k_row = np.zeros(D)
            for d in range(D):
                sq_dist = (X1[n, d] - X2[:, d])**2
                k_row = k_row + np.exp(-1 / (2 * theta**2) * sq_dist)
            if X1_equal_X2:
                k_row[n] = k_row[n] + sigma**2
            result[n] = np.dot(k_row, v)
        return result

    def _conjugate_gradient(self, X, b, theta, sigma, tol, sol0=None):
        """
        Solves the linear system K(X, X; theta) @ sol = b using the conjugate gradient method.

        Parameters:
        - X: np.ndarray, shape (N, D)
            Input points.
        - b: np.ndarray, shape (N,)
            Right-hand side vector.
        - theta: float
            Length-scale parameter for the RBF kernel.
        - sigma: float
            Noise variance.
        - tol: float
            Tolerance for convergence.
        - sol0: np.ndarray, shape (N,), optional
            Initial guess for the solution.

        Returns:
        - sol: np.ndarray, shape (N,)
            Solution to the linear system.
        """
        N = b.shape[0]
        if sol0 is None:
            sol = np.zeros(N)
        else:
            sol = sol0.copy()

        r = b - self._mv_k(X, X, sol, theta, sigma, X1_equal_X2=True)
        p = r.copy()
        r_old = np.dot(r, r)
        max_iter = N

        for i in range(max_iter):
            Kp = self._mv_k(X, X, p, theta, sigma, X1_equal_X2=True)
            alpha = r_old / np.dot(p, Kp)
            sol += alpha * p
            r -= alpha * Kp
            r_new = np.dot(r, r)
            if np.sqrt(r_new) < tol:
                break
            p = r + (r_new / r_old) * p
            r_old = r_new

        print(f"CG converged in {i+1} iterations.")
        return sol

    def set_hyperparameters(self, X, y, theta, sigma):
        """
        Sets the hyperparameters and precomputes the alpha vector for predictions.

        Parameters:
        - X: np.ndarray, shape (N, D)
            Training input points.
        - y: np.ndarray, shape (N,)
            Training target values.
        - theta: float
            Length-scalesr for the RBF kernel.
        - sigma: float
            Noise standard deviation
        """
        self.X = X
        self.y = y
        self.theta = theta
        self.sigma = sigma
        self.alpha = self._conjugate_gradient(X=self.X, b=self.y, theta=self.theta, sigma=self.sigma, tol=0.001)

    def predict(self, X_star):
        """
        Predicts the target values for new input points.

        Parameters:
        - X_star: np.ndarray, shape (N_star, D)
            New input points for prediction.

        Returns:
        - y_star: np.ndarray, shape (N_star,)
            Predicted target values.
        """
        y_star = self._mv_k(X_star, self.X, self.alpha, self.theta, self.sigma, X1_equal_X2=False)
        return y_star
