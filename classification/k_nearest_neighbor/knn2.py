# KNN using sklearn
from movies import training_set, training_labels
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(training_Set, training_labels)

testing_set = [[.45, .2, .5], [.25, .8, .9],[.1, .1, .9]]
result = classifier.predict(testing_set)
print(result)