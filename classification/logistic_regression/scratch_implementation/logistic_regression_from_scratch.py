import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from tabulate import tabulate

class LogisticRegression:
    
    def __init__(self, lr=0.001, epoch=1000, threshold = 0.5):
        """Initilize all the variables

           lr - learning rate
           epoch - numbers of iterations
        """
        self.lr = lr
        self.epoch = epoch
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.log_odds = None
    
    def __sigmoid(self, z):
        """Simulate the sigmoid function"""
        return 1/ (1 + np.exp(-z))

    def __log_odds(self, X):
        """Calculate the return the log odds"""
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        # init parameters
        num_samples, num_features = X.shape
        # init weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # implement gradient descent
        # weights - coefficients
        # bias - intercept
        print("[DEBUG]Training Started...")
        for i in range(self.epoch):
            print(f"[DEBUG]Iterations={i}")
            self.log_odds = self.__log_odds(X)
            y_predicted = self.__sigmoid(self.log_odds)
            
            # Derivatives of weights(coefficient)
            # The derivative of log loss for weights is (Y_predicted - Y) * X
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            # Derivatives of bias(intercept)
            # Derivative of log loss for bias is (Y_predicted -Y)
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            print(f"[DEBUG]weights={self.weights}")
            print(f"[DEBUG]bias={self.bias}")
        print("[DEBUG]Training Completed")

    def predict(self, X):
        self.log_odds = self.__log_odds(X)
        y_predicted = self.__sigmoid(self.log_odds)
        # Apply threshold classification into predicted result
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class

    def predict_proba(self, X):
        self.log_odds = self.__log_odds(X)
        y_predicted = self.__sigmoid(self.log_odds)
        return y_predicted


    def score(self, y_pred, y_true):
        """Calculate and return the accuracy of the model"""
        score = np.sum(y_true == y_pred) / len(y_true)
        return score

if __name__ == '__main__':
    # Use sklearn for data testing
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    dataset_labels = bc.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    def accuracy(y_pred, y_true):
        """Calculate and return the accuracy of the model"""
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    model = LogisticRegression(lr=0.0001, epoch=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print(f"Model Accuracy: {model.score(predictions, y_test)}")

    # Print the data
    y_test = y_test.tolist()
    y_pred = predictions
    print(y_test)
    print(predictions)
