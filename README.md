# 🇩🇿 Sarija Sentiment Analysis — Algerian Darija Sentiment Classifier

This project is an end-to-end **Sentiment Analysis system for Algerian Darija (الدارجة الجزائرية)**  
using a custom dataset of **105,000 labeled samples** split into **train, validation, and test**.

The goal is to build a complete pipeline that cleans raw Algerian text, visualizes the data,  
trains a deep learning model, and provides a real-time prediction API connected to a React interface.

---

## 🧹 Data Preprocessing

A full preprocessing pipeline is implemented to clean Algerian Darija comments.

### ✔ Includes:
- Removing URLs, mentions, emojis, punctuation, and non-Arabic characters  
- Normalizing Arabic letters and removing diacritics  
- Lowercasing text  
- Removing repeated characters (e.g., "راااااائع" → "رائع")  
- Removing Arabic + Algerian stopwords  
- Tokenizing text with Keras Tokenizer (20k vocab)  
- Padding sequences to a fixed length  
- Saving cleaned datasets and tokenizer for training  

The preprocessing script generates:
- cleaned train/val/test datasets  
- `tokenizer.pkl`  
- text statistics  

---

## 📊 Data Visualization (Before & After Preprocessing)

Two analysis scripts visualize the dataset and compare text before and after cleaning.

### Visualizations include:
- WordCloud for raw data  
- WordCloud for cleaned data  
- Class distribution (0 / 1 / 2)  
- Text length distribution  
- Sample comparisons before/after cleaning  

These plots help understand the dataset and verify that preprocessing improves consistency.

---

## 🧠 Model Architecture (CNN + Word2Vec)

The model used is a deep 1D Convolutional Neural Network with pretrained Word2Vec embeddings.

### **Final Architecture:**

- **Embedding Layer**  
  - Word2Vec vectors (200 dimensions)  
  - Trainable
- **SpatialDropout1D (0.2)**
- **Conv1D Layer** (128 filters, kernel=3, ReLU, Same padding, L2 regularization + BatchNorm + Dropout 0.2)
- **Conv1D Layer** (128 filters, kernel=5, ReLU, Same padding, L2 regularization + BatchNorm + Dropout 0.2)
- **GlobalMaxPooling1D**
- **Dense Layer** (64 units, ReLU + BatchNorm + Dropout 0.2)
- **Dense Layer** (32 units, ReLU + BatchNorm + Dropout 0.2)
- **Output Layer** (3 units, Softmax)

### 📌 Training:
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Callbacks: EarlyStopping, ReduceLROnPlateau  
- Accuracy & loss curve plots saved during training  

---

## 🌐 Backend API (Flask)

A lightweight REST API is provided to serve real-time sentiment predictions.


### **Server Features**
- Loads trained `sentiment_cnn_model.h5` and `tokenizer.pkl`  
- Cleans input text using the same preprocessing pipeline as during training  
- Returns both **predicted class** and **confidence score**  
- Handles **single** or **batch predictions**  

## 🖥️ Frontend (React)

The frontend is built with React and provides a user-friendly interface for interacting with the Flask backend.

### **Frontend Features**
- Input box for Algerian Darija comments
- Predict button
- Live display of predicted sentiment and confidence
- Character & word counters
- Mobile responsive layout
- Frontend Repository

## **Frontend Repository**
https://github.com/Ynsaid/Darija_Sentiment_Analysis_Frontend
## **Test live website**
https://darija-sentiment-analysis-frontend-1.onrender.com/

