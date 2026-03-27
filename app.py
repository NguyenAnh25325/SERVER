import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/plant_model_final.keras"
LABELS_PATH = "model/labels.txt"
IMG_SIZE = (224, 224)

model = None
labels = []

# ===== LOAD MODEL =====
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model error:", e)

try:
    with open(LABELS_PATH, "r") as f:
        labels = [x.strip() for x in f]
except Exception as e:
    print("❌ Labels error:", e)


def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]
        img = preprocess(file.read())

        preds = model.predict(img, verbose=0)[0]

        idx = int(np.argmax(preds))
        conf = float(preds[idx])

        label = labels[idx] if idx < len(labels) else str(idx)

        return jsonify({
            "result": f"{label} ({conf*100:.2f}%)"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== IMPORTANT FIX FOR RENDER =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
