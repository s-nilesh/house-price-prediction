import numpy as np
import matplotlib.pyplot as plt
import pickle


class LinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=25):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefs = None
        self.loss = []

    def fit(self, X_train, y_train):
        ones = np.ones((X_train.shape[0], 1))
        X_train_b = np.hstack([ones, X_train])

        self.coefs = (
            np.random.uniform(0, 1, size=(X_train_b.shape[1], 1)) * -0.01
        )  # y = MX + c -- c considered as array of 1 (both weights and biases considered here)
        for _ in range(self.epochs):
            y_pred = self.forward_propagation(X_train_b)
            cost = self.cost_function(y_pred, y_train)
            self.loss.append(cost)
            gradient = self.backward_propagation(X_train_b, y_train, y_pred)
            self.coefs = self.coefs - (self.learning_rate * gradient)

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])
        return X @ self.coefs

    def forward_propagation(self, X_train_b):
        # y = MX
        y_pred = X_train_b @ self.coefs
        return y_pred

    def cost_function(self, y_pred, y_train):
        # mean squared error
        errors = y_pred - y_train.reshape(-1, 1)
        cost = np.mean((errors) ** 2)
        return cost

    def backward_propagation(self, X_train_b, y_train, y_pred):
        errors = y_pred - y_train.reshape(-1, 1)
        gradient = (
            (1 / X_train_b.shape[0]) * X_train_b.T @ errors
        )  # (1/num_rows)*X_transpose @ errors
        return gradient

    def plot_cost(self):
        plt.plot(range(self.epochs), self.loss)

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.coefs, f)
