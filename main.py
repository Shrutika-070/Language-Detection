pip install pandas numpy scikit-learn tensorflow keras matplotlib


import pandas as pd

df = pd.read_csv('/content/language_dataset.csv.zip')
print(df.head())
print(df['Language'].value_counts())

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Encode labels
# convert the "Language" column (text) into numerical labels.
# Stores it in a new column called Language_Code.
label_encoder = LabelEncoder()
df['Language_Code'] = label_encoder.fit_transform(df['Language'])

# Tokenize text
# tokenize it — break it into words and assign each word an integer.
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['Text'])

#Convert text to sequences of numbers
sequences = tokenizer.texts_to_sequences(df['Text'])

# Some sentences are short, some are long. But your LSTM model needs fixed input length.
X = pad_sequences(sequences, maxlen=50)

# Labels
# convert the language labels (e.g., 0, 1, 2) into one-hot encoded vectors
y = pd.get_dummies(df['Language_Code']).values



from sklearn.model_selection import train_test_split

# X = padded input sequences
# y = one-hot encoded language labels

X_train, X_test, y_train, y_test = train_test_split(
    X,         # The padded sequences (input)
    y,         # One-hot encoded labels (output)
    test_size=0.2,  # 20% for testing, 80% for training
    random_state=42 # Ensures reproducibility
)


print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional


# Model Layers Explained
model = Sequential()


#  1. Embedding Layer
# Converts word indices (integers) into dense vectors of size 128.
# input_dim=10000 → Max vocabulary size (same as tokenizer)
# output_dim=128 → Size of each word vector
# input_length=50 → Length of each input sequence (padded to 50)

model.add(Embedding(input_dim=10000, output_dim=128, input_length=50))


# 2. Bidirectional LSTM Layer
# Reads the sentence forward and backward
# Learns better context and improves accuracy
# LSTM(64) means 64 units → larger value = more learning power (but more time to train)

model.add(Bidirectional(LSTM(64)))


# 3. Fully Connected Dense Layer
# Learns non-linear combinations of LSTM outputs
# relu activation helps learn complex patterns

model.add(Dense(64, activation='relu'))


# 4. Output Layer
# y.shape[1] is the number of language classes
# softmax outputs probabilities for each language

model.add(Dense(y.shape[1], activation='softmax'))


 # Compile the Model
 model.compile(
    loss='categorical_crossentropy',  # good for multi-class classification,Best for one-hot encoded multi-class labels
    optimizer='adam',                 # widely used optimizer,Fast, adaptive, and commonly used for NLP tasks
    metrics=['accuracy']              # to track performance, Measures how many predictions are correct (easy to interpret)

)


# Train the Model

history = model.fit(
    X_train, y_train,       # Input and output data to learn from
    epochs=5,               # Number of times the model sees the entire dataset
    batch_size=64,          # Updates weights after every 64 samples
    validation_split=0.2,   # Use 20% of training data for validation,Helps monitor overfitting by checking performance on unseen data
    verbose=1               # Shows training progress in the terminal
)


# Optional: Plot Training Accuracy and Loss
import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()



# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# See Predictions (Optional)
import numpy as np

# Get predicted probabilities
y_pred_probs = model.predict(X_test)

# Convert to class index (e.g., 0 = English, 1 = Hindi, etc.)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)



# Print Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print classification report
print(classification_report(y_true, y_pred))




# Save the Trained Model
model.save("language_detection_model.h5")


# Load the Model (When Needed)
from tensorflow.keras.models import load_model

model = load_model("language_detection_model.h5")


# Make Real-Time Predictions
def predict_language(sentence, tokenizer, model, label_encoder):
    # Preprocess the input
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=50)

    # Predict
    pred = model.predict(padded)
    lang_index = pred.argmax(axis=1)[0]

    # Convert index to label
    predicted_lang = label_encoder.inverse_transform([lang_index])[0]
    return predicted_lang


# Example: Predict a new sentence
sentence = "Je suis étudiant en informatique."
language = predict_language(sentence, tokenizer, model, label_encoder)
print(f"Predicted language: {language}")


# "Where is the nearest hospital?"  [english]
# "¿Dónde está la estación de tren?"  [spanish]
# "यह कैसा काम करता है?"  [hindi]
# "Wie spät ist es?"  [german]
# "Je suis étudiant en informatique."  [french]
