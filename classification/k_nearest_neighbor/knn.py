# Manual implementation of a KNN algorithm from codecademy
# To predict a movie's rating
from movies import movie_dataset, movie_labels, normalize_point

def distance(movie1, movie2):
  """euclidean distance implementation"""
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  """KNN algorithm implementation"""
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

# Check existances
print("Call Me By Your Name" in movie_dataset)

# Data points
my_movie = [350000, 132, 2017]
# Normalize
normalized_my_movie = normalize_point(my_movie)
# Prediction
print(classify(normalized_my_movie, movie_dataset, movie_labels, 5))
