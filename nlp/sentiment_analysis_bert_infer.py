import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

# Enable GPU processing
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Recreate the model architecture
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Build the model (BERT needs to be called on dummy data before loading weights)
dummy_input = {
    "input_ids": tf.constant([[0] * 128]),
    "attention_mask": tf.constant([[0] * 128])
}
_ = bert_model(dummy_input)  # Forward pass to initialize model

# Load saved weights
bert_model.load_weights('bert_sentiment_model_weights.h5')
print("Model weights loaded successfully.")

# Function to preprocess input text
def preprocess_text(text, max_length=128):
    """Tokenizes text and converts it to BERT-compatible input format."""
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        truncation=True
    )
    return {
        "input_ids": tf.constant([inputs['input_ids']]),
        "attention_mask": tf.constant([inputs['attention_mask']])
    }

# Function to make predictions
def predict_sentiment(text):
    """Preprocesses input text, runs inference, and returns sentiment prediction."""
    processed_input = preprocess_text(text)
    logits = bert_model(processed_input).logits
    probabilities = tf.nn.softmax(logits).numpy()[0]

    sentiment = "Positive" if np.argmax(probabilities) == 1 else "Negative"
    confidence = probabilities[np.argmax(probabilities)]

    return sentiment, confidence

# Example inference
sample_review = "This movie was absolutely terrible! The story and acting were lame."
sentiment, confidence = predict_sentiment(sample_review)
print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")
