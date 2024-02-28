# main_script.py
from weights import initialize_weights
from bias import initialize_bias
from perceptron import train, predict


input_size = 3
X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([1, 0])

weights = initialize_weights(input_size)
bias = initialize_bias()


learning_rate = 0.01
epochs = 100
train(weights, bias, X, y, learning_rate, epochs)

predictions = predict(weights, bias, X)
print("Predictions:", predictions)
