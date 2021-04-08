# This is an implementation from scratch of the logistic regression

# Algorithm of logistic regression
# The main goal of logistic regression is to find the best coefficients, and intercept
# to make the prediction accurate
import numpy as np
from exam import hours_studied, passed_exam

# For some reason np array that start with an int will only be an int
coefficients = np.array([[1.1]])
intercept = np.array([1.0])
learning_rate = 0.0001

def log_odds(features, coefficients, intercept):
  """Return the log odds for all the features"""
  return np.dot(features,coefficients) + intercept

def sigmoid(z):
    """Simulate the sigmoid function and return a result"""
    denominator = 1 + np.exp(-z)
    return 1/denominator

def log_loss(probabilities, actual_class, Xi):
    """Return a summed log loss of all values in the feature dataset"""
    return np.sum(-(1/actual_class.shape[0])*(actual_class*np.log(probabilities) + (1-actual_class)*np.log(1-probabilities))*Xi)

def get_error_rate(predicted, actual):
    """Return the average error rate"""
    error = 0
    for i in range(len(predicted)):
        error += abs(predicted[i][0] - actual[i][0])
    return error/len(predicted) 

def train(features, labels):
    n = len(features)
    error_rate = 1
    i = 0
    print("Training Started!")
    while error_rate > 0:
        print(f"Iteration={i}")
        # First get the probabilities
        calculated_log_odds = log_odds(features, coefficients, intercept)
        probabilities = sigmoid(calculated_log_odds)
        # Calculate log loss
        logloss = log_loss(probabilities, labels, features)
        # Apply gradient descent algorithm
        derivative_sigmoid = (1/n) * logloss
        coefficients[0][0] = coefficients[0][0] - learning_rate*derivative_sigmoid
        intercept[0] = intercept[0] - learning_rate*derivative_sigmoid
        print(f"coefficient={coefficients},intercept={intercept}")
        # Calculate error rate
        result = np.where(probabilities >= 0.5, 1, 0)
        old_error_rate = error_rate
        error_rate = get_error_rate(result,labels)
        print(f"[debug]error_rate:{error_rate}")
        i += 1

# Create predict_class() function
def predict(features, threshold):
  calculated_log_odds = log_odds(features, coefficients, intercept)
  probabilities = sigmoid(calculated_log_odds)
  result = np.where(probabilities >= threshold, 1, 0)
  return result

# First, train the model
train(hours_studied, passed_exam)

# Make final classifications on Codecademy University data here
final_results = predict(hours_studied, 0.5)
predicted = final_results.tolist()
print(f"predicted: {predicted}")
actual = passed_exam.tolist()
print(f"realY_val: {actual}")
