# Actual movies.py from codecademy
import pandas as pd
from random import shuffle, seed
import numpy as np

seed(100)

df = pd.read_csv("movies.csv")
df = df.dropna()

good_movies = df.loc[df['imdb_score'] >= 7]
bad_movies = df.loc[df['imdb_score'] < 7]

def min_max_normalize(lst):
	minimum = min(lst)
	maximum = max(lst)
	normalized = []

	for value in lst:
		normalized_num = (value - minimum) / (maximum - minimum)
		normalized.append(normalized_num)

	return normalized

x_good = good_movies["budget"]
y_good = good_movies["duration"]
z_good = good_movies['title_year']
x_bad = bad_movies["budget"]
y_bad = bad_movies["duration"]
z_bad = bad_movies['title_year']

data = [x_good, y_good, z_good, x_bad, y_bad, z_bad]
arrays_data = []

for d in data:
  norm_d = min_max_normalize(d)
  arrays_data.append(np.array(norm_d))


good_class = list(zip(arrays_data[0].flatten(), arrays_data[1].flatten(), arrays_data[2].flatten(),(np.array(([1] * len(arrays_data[0])))) ))
bad_class = list(zip(arrays_data[3].flatten(), arrays_data[4].flatten(), arrays_data[5].flatten(),(np.array(([0] * len(arrays_data[0])))) ))

dataset = good_class + bad_class
shuffle(dataset)

movie_dataset = []
labels = []
for movie in dataset:
  movie_dataset.append(movie[:-1])
  labels.append(movie[-1])
