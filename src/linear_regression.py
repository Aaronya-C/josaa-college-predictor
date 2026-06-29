import numpy as np

class LinearRegression:
    """
    Linear Regression from scratch using the Normal Equation.
    Based on CS229 lecture notes.
    theta = (X^T X)^{-1} X^T y
    """

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Fit model using Normal Equation.
        X: numpy array of shape (m, n)
        y: numpy array of shape (m,)
        """
        # Add bias column (column of ones) — this is x_0 = 1
        X_b = np.hstack([np.ones((len(X), 1)), X])

        # Normal equation: theta = pinv(X^T X) @ X^T @ y
        # We use pinv (pseudoinverse) instead of inv for numerical stability
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        """
        Predict output for input X.
        X: numpy array of shape (m, n)
        """
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b @ self.theta

    def rmse(self, X, y):
        """
        Root Mean Squared Error — tells us how far off predictions are on average.
        """
        predictions = self.predict(X)
        return np.sqrt(np.mean((predictions - y) ** 2))