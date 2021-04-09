# A single perceptron network that simulate an OR gate
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Data set of different logic gates
data = [[0,0],[0,1],[1,0],[1,1]]
labels = [0,1,1,1]

# Scatter Plot of Data
x = [i[0] for i in data]
y = [i[1] for i in data]
c = [i for i in labels]
plt.scatter(x,y,c=labels)

# ML Model
classifier = Perceptron(max_iter=40)
classifier.fit(data, labels)

# Get score
print(classifier.score(data, labels))

# Decision Function
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

# Set Up Heat Map
x_values = np.linspace(0,1,100)
y_values = np.linspace(0,1,100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(i) for i in distances]
abs_distances_2d = np.reshape(abs_distances, (100,100))

# Draw Map
heatmap = plt.pcolormesh(x_values, y_values, abs_distances_2d)
plt.colorbar(heatmap)

# predict
x_test = [[0,0],[1,1],[1,0],[1,0],[0,0]]
y_test = [0,1,1,1,0]
print(classifier.predict(x_test))
print(classifier.score(x_test, y_test))


plt.show()
