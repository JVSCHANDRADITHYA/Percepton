import numpy as np
from WandB.weights import initialize_weights
from WandB.bias import initialize_bias

def train(weights, bias, X, y, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            prediction = predict(weights, bias, xi)
            update = learning_rate * (target - prediction)
            weights += update * xi
            bias += update

def predict(weights, bias, X):
    z = np.dot(X, weights) + bias
    return np.where(z >= 0, 1, 0)