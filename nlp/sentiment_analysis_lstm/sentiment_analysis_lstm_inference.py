import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load tokenizer
tokenizer_file = "tokenizer.pkl"
with open(tokenizer_file, 'rb') as f:
    tokenizer = pickle.load(f)

# Define model parameters (must match training)
max_features = 10000
max_length = 100

# Recreate the model architecture
model = Sequential([
    Embedding(input_dim=max_features, output_dim=100, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Build the model before loading weights
model.build(input_shape=(None, max_length))

# Load saved weights
model.load_weights('lstm_sentiment_model.weights.h5')
print("Model weights loaded successfully.")

# Function to preprocess input text
def preprocess_spacy(text):
    """Lemmatizes and removes stopwords/punctuation using SpaCy."""
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

# Function to make predictions
def predict_sentiment(text):
    """Preprocesses input text, converts to sequence, and predicts sentiment."""
    processed_text = preprocess_spacy(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    prediction = model.predict(padded_sequence)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if sentiment == "Positive" else 1 - prediction

    return sentiment, confidence

# Example inference
sample_review = "This movie was absolutely fantastic! The story and acting were amazing."
sentiment, confidence = predict_sentiment(sample_review)
print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")
