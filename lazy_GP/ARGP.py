from .GP import GP
import numpy as np

class ARGP(GP):

    def _prepare_arx_data(self, X, Y, N_AR):
        """
        Prepares the ARX (Auto-Regressive with eXogenous inputs) data matrix.

        Parameters:
        X (numpy.ndarray): Exogenous input data of shape (N, D).
        Y (numpy.ndarray): Target data of shape (N,).
        N_AR (int): autoregressive order

        Returns:
            - X_hat (numpy.ndarray): Combined AR and exogenous features.
            - Y_hat (numpy.ndarray): Target values as a column vector.
        """

        self.N_AR = N_AR

        # Initialise auto-regressive features, exogenous features and targets
        ar_features, exog_features, targets = [], [], []

        # Create ARX data-matrix
        for t in range(self.N_AR, len(Y)):
            ar_features.append(Y[t-self.N_AR:t])     # AR terms
            exog_features.append(X[t])       # Exogenous inputs
            targets.append(Y[t])             # Target value
        X_hat = np.hstack([exog_features, ar_features])
        Y_hat = np.array(targets)

        return X_hat, Y_hat
    
    def set_hyperparameters(self, X, y, theta, sigma, N_AR, tol=0.001):
        X_AR, y_AR = self._prepare_arx_data(X, y, N_AR)
        super().set_hyperparameters(X=X_AR, y=y_AR, theta=theta, sigma=sigma, tol=tol)

    def predict_full_model(self, X_star, y0):
        """
        Full model predictions
        """

        assert len(y0) == self.N_AR

        Y = []
        D = np.shape(X_star)[1]
        for t in range(self.N_AR, np.shape(X_star)[0] + self.N_AR):

            # First time step
            if t == self.N_AR:
                x = np.hstack([X_star[0], y0])
                
            # Remaining time steps
            else:
                x[:D] = X_star[t - self.N_AR]
                x[D:] = np.roll(x[D:], 1)
                x[-1] = y

            y = super().predict(x.reshape(1, -1))
            Y.append(y)

        # Finish by converting Y to array
        Y = np.array(Y)

        return Y
