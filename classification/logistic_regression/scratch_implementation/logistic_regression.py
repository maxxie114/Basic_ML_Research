# This is an implementation from scratch of the logistic regression
# Algorithm, but it is untrained, and the gradient descent
# algorithm and log loss for training still need to be 
# implemented
import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

def log_odds(features, coefficients,intercept):
  return np.dot(features,coefficients) + intercept

def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1/denominator

# Create predict_class() function here
def predict_class(features, coefficients, intercept, threshold):
  calculated_log_odds = log_odds(features, coefficients, intercept)
  probabilities = sigmoid(calculated_log_odds)
  result = np.where(probabilities >= threshold, 1, 0)
  return result

# Make final classifications on Codecademy University data here
final_results = predict_class(hours_studied, calculated_coefficients, intercept, 0.5)
print(final_results)
