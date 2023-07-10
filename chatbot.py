import tensorflow as tf
from tensorflow.keras import layers

# Define the dataset for training the chatbot
questions = ["What is your name?", "How are you?", "What is the weather today?"]
answers = ["My name is Chatbot.", "I'm doing well.", "The weather is sunny."]

# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences of tokens
input_sequences = tokenizer.texts_to_sequences(questions)
output_sequences = tokenizer.texts_to_sequences(answers)

# Pad sequences to have the same length
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences)
output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences)

# Define the chatbot model
model = tf.keras.models.Sequential([
    layers.Embedding(total_words, 100, input_length=input_sequences.shape[1]),
    layers.Bidirectional(layers.GRU(64, return_sequences=True)),
    layers.GRU(64),
    layers.Dense(total_words, activation="softmax")
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the chatbot model
model.fit(input_sequences, output_sequences, epochs=50)

# Generate responses from the trained model
def generate_response(user_input):
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_sequence = tf.keras.preprocessing.sequence.pad_sequences(user_input_sequence, maxlen=input_sequences.shape[1])
    predicted_output = model.predict(user_input_sequence)
    predicted_index = tf.argmax(predicted_output, axis=-1).numpy()[0]
    return tokenizer.index_word[predicted_index]

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = generate_response(user_input)
    print("Chatbot:", response)
