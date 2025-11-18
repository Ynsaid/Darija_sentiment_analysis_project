# 🇩🇿 Sarija Sentiment Analysis — Algerian Darija Sentiment Classifier

This project is an end-to-end **Sentiment Analysis system for Algerian Darija (الدارجة الجزائرية)**  
using a custom dataset of **105,000 labeled samples** split into **train, validation, and test**.

The goal is to build a complete pipeline that cleans raw Algerian text, visualizes the data,
trains a deep learning model, and provides a real-time prediction API connected to a React interface.

---

## 🧹 Data Preprocessing

A full preprocessing pipeline is implemented to clean Algerian Darija comments:

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
- tokenizer.pkl  
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

## 🧠 Model Training — CNN + Word2Vec

The model is trained using a hybrid architecture:

### ✔ Word2Vec embedding layer  
Trained directly on the 105k Algerian Darija dataset to learn local dialect expressions.

### ✔ CNN architecture  
- 1D Convolution  
- Global Max Pooling  
- Dropout + Dense layers  
- Softmax output for 3 sentiment classes  

### ✔ Training features:
- EarlyStopping  
- ReduceLROnPlateau  
- Accuracy & loss curve visualization  
- Saving best model (`sentiment_cnn_model.h5`)  

The result is a robust sentiment classifier adapted to Darija vocabulary.

---

## 🌐 Backend API (Flask)

A lightweight REST API is provided to serve real-time sentiment predictions.

### Endpoint:
