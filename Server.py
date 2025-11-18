from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)
CORS(app, origins=["https://darija-sentiment-analysis-frontend-1.onrender.com"])

H5_FILE_PATH = r'sentiment_cnn_model.h5'
PKL_FILE_PATH =r'tokenizer.pkl'


try:
    model = load_model(H5_FILE_PATH)
    with open(PKL_FILE_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or tokenizer: {e}")

MAX_LEN = 100  
labels = ['negative', 'neutral', 'positive']  


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({"error": "الرجاء إرسال 'text' في جسم الطلب"}), 400

        text_to_analyze = data['text']

       
        seq = tokenizer.texts_to_sequences([text_to_analyze])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

     
        pred_probs = model.predict(padded)[0]  
        predicted_index = pred_probs.argmax()
        predicted_label = labels[predicted_index]

  
        confidences_dict = {labels[i]: float(pred_probs[i]) for i in range(len(labels))}

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": confidences_dict,
            "text": text_to_analyze
        })

    except Exception as e:
        print(f"!!! خطأ في /predict: {e}")
        return jsonify({
            "status": "error",
            "message": f"حدث خطأ في الخادم: {str(e)}"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host='0.0.0.0', port=port, debug=False)