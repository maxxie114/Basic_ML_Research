import os
import pickle
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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

# Create PyTorch Dataset and DataLoader
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = IMDBDataset(padded_train, y_train_subset)
test_dataset = IMDBDataset(padded_test, y_test_subset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out).squeeze()

# Initialize model, loss function, and optimizer
model = SentimentLSTM(vocab_size=max_features, embed_dim=100, hidden_dim=64).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        predictions = (outputs >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
print(f"Accuracy: {correct/total:.4f}")

# Save model weights
torch.save(model.state_dict(), 'lstm_sentiment_model.pth')
print("Model weights saved.")
