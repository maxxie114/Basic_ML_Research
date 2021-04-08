import numpy as np
from sklearn.linear_model import LogisticRegression
from exam import hours_studied_scaled, passed_exam, exam_features_scaled_train, exam_features_scaled_test, passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled

# Create and fit logistic regression model here
model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)

# Save the model coefficients and intercept here
calculated_coefficients = model.coef_
intercept = model.intercept_


# Predict the probabilities of passing for next semester's students here
# This predict probabilities
passed_predictions = model.predict_proba(guessed_hours_scaled)

# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled_train, passed_exam_2_train)

# Predict whether the students will pass here
# This output a yes or no
passed_predictions_2 = model_2.predict(exam_features_scaled_test)
print(passed_predictions_2)
