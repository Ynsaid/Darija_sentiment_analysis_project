import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Dropout ,SpatialDropout1D ,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

#load datasets
train_df = pd.read_csv(r"train_clean.csv")
val_df   = pd.read_csv(r"val_clean.csv")
test_df  = pd.read_csv(r"test_clean.csv")
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)



# Prepare data
X_train, y_train = train_df["text"].astype(str).values, train_df["label"].values
X_val,   y_val   = val_df["text"].astype(str).values, val_df["label"].values
X_test,  y_test  = test_df["text"].astype(str).values, test_df["label"].values


#Tokenization & Padding

MAX_VOCAB = 20000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
X_val_seq   = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=MAX_LEN, padding='post')
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

# Save tokenizer for frontend
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved successfully.")


#Build CNN Model
model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=128, input_length=MAX_LEN),
    SpatialDropout1D(0.3),

    # Conv layers with BN; L2 reduced
    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    Conv1D(128, 5, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),

    GlobalAveragePooling1D(),

    # Dense layers reduced size
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(3, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


#Train Model 
early_stop = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)



history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


# Evaluate Model
y_pred = np.argmax(model.predict(X_test_seq), axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Accuracy and Loss Curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')
plt.show()

# Save Model
model.save("sentiment_cnn_model.h5")
print("✅ Model saved successfully at sentiment_cnn_model.h5")
