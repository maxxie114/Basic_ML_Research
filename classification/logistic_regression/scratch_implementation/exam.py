# Code and data from Codecademy
import numpy as np

hours_studied = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).reshape(20,1)

passed_exam = np.array([0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1]).reshape(20,1)

math_courses_taken = np.array([0,1,4,0,3,1,2,3,0,2,0,1,2,3,4,2,1,3,1,0]).reshape(20,1)

calculated_coefficients = np.array([[0.20678491]])

zero_coefficients = np.array([0])

intercept = np.array([-1.76125712])

zero_intercept = np.array([0])

def log_odds(features, coefficients,intercept):
  return np.dot(features,coefficients) + intercept

calculated_log_odds = log_odds(hours_studied,calculated_coefficients,intercept)

calculated_log_odds_2 = log_odds(hours_studied,zero_coefficients,zero_intercept)

def sigmoid(z):
  denominator = 1 + np.exp(-z)
  return 1/denominator

probabilities = sigmoid(calculated_log_odds)

probabilities_2 = sigmoid(calculated_log_odds_2)

def log_loss(probabilities,actual_class):
  return np.sum(-(1/actual_class.shape[0])*(actual_class*np.log(probabilities) + (1-actual_class)*np.log(1-probabilities)))
