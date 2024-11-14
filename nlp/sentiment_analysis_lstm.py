import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from multiprocessing import Pool, cpu_count
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

# Enable GPU processing
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Load IMDB dataset
max_features = 10000  # Number of words to consider as features
max_length = 100  # Cut texts after this number of words (among top max_features most common words)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Read the content of one of the data
word_index = imdb.get_word_index()
index_to_word = {index: word for word, index in word_index.items()}

def decode_review(encoded_review):
  decoded_review = " ".join([index_to_word.get(i - 3, '') for i in encoded_review])
  return decoded_review

# Example usage
decoded_review = decode_review(x_train[0])
# print(decoded_review)

# Decode function to convert integers back to words
word_index = imdb.get_word_index()
index_to_word = {v: k for k, v in word_index.items()}

# Convert indices back to words for preprocessing with SpaCy
def decode_review(encoded_review):
    return ' '.join(index_to_word.get(i - 3, '?') for i in encoded_review if i >= 3)

# Preprocess a single review with SpaCy
def preprocess_spacy(text):
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

# Wrapper function for parallel processing
def preprocess_review(encoded_review):
    decoded = decode_review(encoded_review)
    preprocessed = preprocess_spacy(decoded)
    return preprocessed

# Preprocess dataset in parallel using Pool
def preprocess_dataset_parallel(dataset, num_workers=cpu_count()):
    with Pool(num_workers) as pool:
        return pool.map(preprocess_review, dataset)

# Preprocess training and testing data
print("Preprocessing the data...")
preprocessed_train = preprocess_dataset_parallel(x_train[:100000])  # Subset for demonstration
preprocessed_test = preprocess_dataset_parallel(x_test[:10000])  # Subset for demonstration

# Tokenization and padding using Keras Tokenizer
tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(preprocessed_train)
sequences_train = tokenizer.texts_to_sequences(preprocessed_train)
sequences_test = tokenizer.texts_to_sequences(preprocessed_test)
padded_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
padded_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')

y_train_subset = y_train[:100000]
x_train_subset = x_train[:100000]
y_test_subset = y_test[:10000]
x_test_subset = x_test[:10000]

### Step 3: Define and Train the Model
model = Sequential([
    Embedding(input_dim=max_features, output_dim=100, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model 2
model.fit(padded_train, y_train_subset, epochs=10, batch_size=32, validation_split=0.2)

print(model.summary())

# Evaluate model 2
loss, accuracy = model.evaluate(padded_test, y_test_subset)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# After training the model, save the model weights
model.save_weights('lstm_sentiment_model.weights.h5')
print("Model weights saved.")
