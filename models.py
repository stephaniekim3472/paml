import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.b = None

    def fit(self, X, y):
        n, d = X.shape
        X_b = np.hstack([X, np.ones((n, 1))])
        I = np.eye(d + 1)
        I[-1, -1] = 0
        self.weights = np.linalg.solve(
            X_b.T @ X_b + self.alpha * I,
            X_b.T @ y
        )
        self.w = self.weights[:-1]
        self.b = self.weights[-1]
        return self

    def predict(self, X):
        return X @ self.w + self.b


class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict(self, X, batch_size=500):
        n = X.shape[0]
        predictions = np.empty(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            sq_X = np.sum(X_batch ** 2, axis=1, keepdims=True)
            sq_train = np.sum(self.X_train ** 2, axis=1)
            dists = sq_X + sq_train - 2 * X_batch @ self.X_train.T
            dists = np.maximum(dists, 0)
            knn_idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]
            for i in range(end - start):
                predictions[start + i] = self.y_train[knn_idx[i]].mean()
        return predictions


class MLPRegressor:
    def __init__(self, input_dim, hidden_dim=64, lr=0.001, alpha=0.001,
                 batch_size=256, epochs=100, patience=10, verbose=True):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)
        self.train_losses = []
        self.val_losses = []

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_grad(self, z):
        return (z > 0).astype(np.float64)

    def _forward(self, X):
        self._z1 = X @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        return self._z2.flatten()

    def _backward(self, X, y, y_pred):
        n = X.shape[0]
        d_z2 = (y_pred - y).reshape(-1, 1) / n
        dW2 = self._a1.T @ d_z2 + self.alpha * self.W2
        db2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * self._relu_grad(self._z1)
        dW1 = X.T @ d_z1 + self.alpha * self.W1
        db1 = d_z1.sum(axis=0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, X_val=None, y_val=None):
        n = X.shape[0]
        best_val_loss = np.inf
        best_weights = None
        wait = 0
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            epoch_loss, n_batches = 0, 0
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                y_pred = self._forward(X_shuf[start:end])
                epoch_loss += np.mean((y_pred - y_shuf[start:end]) ** 2)
                n_batches += 1
                self._backward(X_shuf[start:end], y_shuf[start:end], y_pred)
            self.train_losses.append(epoch_loss / n_batches)
            if X_val is not None:
                val_loss = np.mean((self._forward(X_val) - y_val) ** 2)
                self.val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = (self.W1.copy(), self.b1.copy(),
                                    self.W2.copy(), self.b2.copy())
                    wait = 0
                else:
                    wait += 1
                if wait >= self.patience:
                    break
        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights
        return self

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        return (a1 @ self.W2 + self.b2).flatten()
