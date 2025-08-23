# 🌍 Language Detection using Deep Learning

This project implements a **Language Detection model** that predicts the language of a given sentence. It uses **LSTM (Long Short-Term Memory) Neural Networks** for text classification.

---

## 🚀 Features
- Detects the language of a text input  
- Uses **Bidirectional LSTM** for better context learning  
- Works on multiple languages (English, French, Hindi, German, Spanish, etc.)  
- Evaluates with **Confusion Matrix** and **Classification Report**  
- Provides a simple prediction function for real-time usage  

---

## 🧠 Tech Stack
- **Python**  
- **TensorFlow / Keras** (Deep Learning)  
- **scikit-learn** (Label Encoding, Train/Test split)  
- **NumPy & Pandas** (Data processing)  
- **Matplotlib & Seaborn** (Visualization)  

---

## 📂 Project Workflow
1. **Data Preprocessing**
   - Load dataset (`Text`, `Language`)  
   - Encode languages → numbers using **LabelEncoder**  
   - Tokenize and pad text sequences  

2. **Model Building**
   - **Embedding Layer** → Convert words into vectors  
   - **Bidirectional LSTM** → Learn sequential patterns both forward & backward  
   - **Dense Layers** → Learn complex patterns  
   - **Softmax Output** → Predicts probabilities for each language  

3. **Training & Evaluation**
   - Train using **Categorical Crossentropy** loss and **Adam Optimizer**  
   - Evaluate with Accuracy, Confusion Matrix & Classification Report  

4. **Saving the Model**
   - Model is saved as `language_detection_model.h5`  
   - Tokenizer & Label Encoder can be saved separately (for deployment)  

5. **Prediction Function**
   - Takes a new sentence  
   - Tokenizes & pads it  
   - Predicts the language  

---

## ⚡ Example Usage
```python
sentence = "Je suis étudiant en informatique."
language = predict_language(sentence, tokenizer, model, label_encoder)
print(f"Predicted language: {language}")

Predicted language: French

