import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import create_optimizer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# Enable GPU processing
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=512):
    input_ids, attention_masks, labels = [], [], []

    for text, label in examples:
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True
        )
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
        labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": tf.constant(input_ids),
            "attention_mask": tf.constant(attention_masks)
        },
        tf.constant(labels)
    ))
    return dataset

# Load IMDB dataset
max_length = 128  # Suitable length for BERT
print("Loading IMDB Datasets...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=1000)
word_index = tf.keras.datasets.imdb.get_word_index()
index_to_word = {i + 3: word for word, i in word_index.items()}
index_to_word[0], index_to_word[1], index_to_word[2] = '<PAD>', '<START>', '<UNK>'

def decode_review(encoded_review):
    return ' '.join([index_to_word.get(i, '?') for i in encoded_review])

# Limit to 1000 sentences for training
num_train_samples = 25000

# Prepare a small subset of examples for training
print("Preparing a subset of the training dataset...")
train_examples = [(decode_review(x), y) for x, y in zip(x_train[:num_train_samples], y_train[:num_train_samples])]
test_examples = [(decode_review(x), y) for x, y in zip(x_test, y_test)]  # Use full test dataset

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Convert data to BERT inputs
train_dataset = convert_examples_to_tf_dataset(train_examples, tokenizer, max_length=max_length).shuffle(1000).batch(16)
test_dataset = convert_examples_to_tf_dataset(test_examples, tokenizer, max_length=max_length).batch(16)

# Define number of training steps
num_train_steps = len(train_dataset) * 3  # For example, 3 epochs

# Create an optimizer
optimizer, _ = create_optimizer(
    init_lr=3e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=0,
    weight_decay_rate=0.01
)

# Compile model
print("Compiling model...")
bert_model.compile(
    optimizer=optimizer,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print(bert_model.summary())

# Train the model
print("Training model with a subset of the dataset...")
bert_model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Save the model weights after training
bert_model.save_weights('bert_sentiment_model_weights.h5')

# Evaluate model
loss, accuracy = bert_model.evaluate(test_dataset)
print(f'Loss: {loss}, Accuracy: {accuracy}')
