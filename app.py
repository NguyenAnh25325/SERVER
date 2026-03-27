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

# ─── Config ───────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model/plant_model_final.keras")
LABELS_PATH = os.environ.get("LABELS_PATH", "model/labels.txt")
IMG_SIZE = (224, 224)

model = None
labels = []

# ─── Load model ───────────────────────────────────────────
print("⏳ Loading model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded!")
except Exception as e:
    print("❌ Model load error:", str(e))

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(labels)} labels")
except Exception as e:
    print("❌ Labels load error:", str(e))


# ─── Image preprocess ─────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


# ─── ROUTES ───────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "labels": len(labels)
    })


# ─── PREDICT (GỌN NHẤT THEO YÊU CẦU) ─────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        img_array = preprocess_image(file_bytes)
        preds = model.predict(img_array, verbose=0)[0]

        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        label = labels[idx]

        # ✔ OUTPUT GỌN NHẤT
        return jsonify({
            "result": f"{label} ({confidence * 100:.2f}%)"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ─── RUN ───────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
