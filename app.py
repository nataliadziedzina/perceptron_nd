# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

X = np.array([
    [2, 3],
    [1, 1],
    [2, 1],
    [3, 4],
    [3, 2],
    [1, 3]
])
y = np.array([1, 0, 0, 1, 1, 0])


perceptron = Perceptron(learning_rate=0.1, n_iters=1000)
perceptron.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_new = np.array(data['input'])
    prediction = perceptron.predict(X_new)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/')
def say_hello():
    return "Hello World"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
