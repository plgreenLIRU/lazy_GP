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

        N1, D = X1.shape
        N2, D = X2.shape
        result = np.zeros(N1)
        for n in range(N1):
            k_row = np.zeros(N2)
            for d in range(D):
                sq_dist = (X1[n, d] - X2[:, d].T)**2
                k_row = k_row - 1 / (2 * theta[d]**2) * sq_dist
            k_row = np.exp(k_row)
            if X1_equal_X2:
                k_row[n] = k_row[n] + sigma**2
            result[n] = np.dot(k_row, v)
        return result

    def _mv_dk(self, X, d_dash, v, theta):
        """
        Lazily computes the derivative of the kernel matrix with respect to a specific 
        length-scale parameter (theta[d_dash]) and multiplies it by a vector v.

        Parameters:
        - X: np.ndarray, shape (N, D)
            Input points.
        - d_dash: int
            Index of the length-scale parameter with respect to which the derivative is computed.
        - v: np.ndarray, shape (N,)
            Vector to multiply with the derivative of the kernel matrix.
        - theta: np.ndarray, shape (D,)
            Length-scale parameters for the RBF kernel.

        Returns:
        - result: np.ndarray, shape (N,)
            Result of the matrix-vector multiplication with the derivative of the kernel matrix.
        """

        N, D = X.shape
        result = np.zeros(N)    
        for i in range(N):
            dk_row = np.zeros(N)
            for d in range(D):
                dk_row = dk_row - 0.5 * (X[i, d] - X[:, d])**2
            dk_row = np.exp(dk_row)
            dk_row = dk_row * theta[d_dash]**-3 * (X[i, d_dash] - X[:, d_dash].T)**2
            result[i] = np.dot(dk_row, v)
        return result

    def _tr_invK_dK(self, X, theta, sigma, d_dash, S=10):
        """
        Approximates the trace of the product of the inverse kernel matrix and 
        the derivative of the kernel matrix with respect to a specific length-scale parameter.
        Is needed as part of evaluating the gradient of the log-likelihood.

        Parameters:
        - X: np.ndarray, shape (N, D)
            Input points.
        - theta: np.ndarray, shape (D,)
            Length-scale parameters for the RBF kernel.
        - sigma: float
            Noise standard deviation.
        - d_dash: int
            Index of the length-scale parameter with respect to which the derivative is computed.
        - S: int, optional (default=10)
            Number of Monte Carlo samples used for the approximation.

        Returns:
        - final_tr_term: float
            Approximated trace term.
        """        
        N = X.shape[0]
        final_tr_term = 0
        for i in range(S):
            z = np.random.randn(N)
            q = self._conjugate_gradient(X=X, b=z, theta=theta, sigma=sigma)

            K, C, inv_C, dK_dtheta = self._find_exact_matrices(X=X, theta=theta, sigma=sigma, d_dash=d_dash)
            #final_tr_term = final_tr_term + z @ inv_C @ dK_dtheta @ z
            #final_tr_term = final_tr_term + z @ inv_C @ self._mv_dk(X=X, d_dash=d_dash, v=z, theta=theta)
            final_tr_term = final_tr_term + q @ self._mv_dk(X=X, d_dash=d_dash, v=z, theta=theta)
  
        final_tr_term /= S
        return final_tr_term

    def _find_exact_matrices(self, X, theta, sigma, d_dash):
        # Only used for tests

        N, D = np.shape(X)
        K = np.ones([N, N])
        dK_dtheta = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                dK_dtheta[i, j] = theta[d_dash]**-3
                for d in range(D):
                    K[i, j] *= np.exp(-1 / (2 * theta[d]**2) * (X[i, d] - X[j, d])**2)
                    dK_dtheta[i, j] *= np.exp(-0.5 * (X[i, d] - X[j, d])**2)
                dK_dtheta[i, j] *= (X[i, d_dash] - X[j, d_dash])**2
        C = K + np.eye(N) * sigma**2
        inv_C = np.linalg.inv(C)
        return K, C, inv_C, dK_dtheta

    def _conjugate_gradient(self, X, b, theta, sigma, sol0=None, verbose=True):
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

        #tol = 1e-9

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
            #if np.sqrt(r_new) < tol:
            #    break
            p = r + (r_new / r_old) * p
            r_old = r_new
        
        if verbose:
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
        self.alpha = self._conjugate_gradient(X=self.X, b=self.y, theta=self.theta, sigma=self.sigma)

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
