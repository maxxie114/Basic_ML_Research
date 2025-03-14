import os
import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Enable GPU processing
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Load IMDB dataset
max_features = 10000  # Number of words to consider as features
max_length = 100  # Cut texts after this number of words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Define file paths for saving preprocessed data
train_data_file = "preprocessed_train.pkl"
test_data_file = "preprocessed_test.pkl"
tokenizer_file = "tokenizer.pkl"

# Function to decode reviews
word_index = imdb.get_word_index()
index_to_word = {index: word for word, index in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([index_to_word.get(i - 3, '?') for i in encoded_review if i >= 3])

# Preprocess with SpaCy
def preprocess_spacy(text):
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

# Load preprocessed data if available, otherwise preprocess
if os.path.exists(train_data_file) and os.path.exists(test_data_file):
    print("Loading preprocessed data...")
    with open(train_data_file, 'rb') as f:
        preprocessed_train = pickle.load(f)
    with open(test_data_file, 'rb') as f:
        preprocessed_test = pickle.load(f)
else:
    print("Preprocessing the data...")
    preprocessed_train = [preprocess_spacy(decode_review(review)) for review in x_train[:100000]]
    preprocessed_test = [preprocess_spacy(decode_review(review)) for review in x_test[:10000]]
    
    # Save preprocessed data
    with open(train_data_file, 'wb') as f:
        pickle.dump(preprocessed_train, f)
    with open(test_data_file, 'wb') as f:
        pickle.dump(preprocessed_test, f)

# Tokenization and padding
if os.path.exists(tokenizer_file):
    print("Loading tokenizer...")
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
else:
    print("Training tokenizer...")
    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
    tokenizer.fit_on_texts(preprocessed_train)
    
    # Save tokenizer
    with open(tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)

sequences_train = tokenizer.texts_to_sequences(preprocessed_train)
sequences_test = tokenizer.texts_to_sequences(preprocessed_test)
padded_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
padded_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')

# Reduce dataset size
y_train_subset = y_train[:100000]
y_test_subset = y_test[:10000]

# Define and train the model
model = Sequential([
    Embedding(input_dim=max_features, output_dim=100, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(padded_train, y_train_subset, epochs=10, batch_size=32, validation_split=0.2)

print(model.summary())

# Evaluate the model
loss, accuracy = model.evaluate(padded_test, y_test_subset)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save model weights
model.save_weights('lstm_sentiment_model.weights.h5')
print("Model weights saved.")
