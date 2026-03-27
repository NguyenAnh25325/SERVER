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

# ─── Load model & labels once at startup ──────────────────
print("⏳ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]
print(f"✅ Loaded {len(labels)} labels: {labels}")


# ─── Helper ───────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → preprocessed numpy array for EfficientNet."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)
    arr = preprocess_input(arr)          # EfficientNet-specific scaling
    return arr


# ─── Routes ───────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "Tomato Disease Detection API",
        "endpoints": {
            "POST /predict": "Upload an image file (field name: 'file')",
            "GET  /labels":  "Get list of all class labels",
            "GET  /health":  "Health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/labels", methods=["GET"])
def get_labels():
    return jsonify({"labels": labels, "count": len(labels)})


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate request ──────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file field in request. Use multipart/form-data with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type and file.content_type not in allowed_types:
        return jsonify({"error": f"Unsupported content type: {file.content_type}"}), 415

    # ── Inference ─────────────────────────────────────────
    try:
        file_bytes = file.read()
        img_array = preprocess_image(file_bytes)
        preds = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # ── Build response ────────────────────────────────────
    result_idx = int(np.argmax(preds))
    confidence = float(preds[result_idx])
    predicted_label = labels[result_idx]
    is_healthy = "healthy" in predicted_label.lower()

    # Top-3 predictions
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        {"label": labels[i], "confidence": float(preds[i])}
        for i in top3_idx
    ]

    return jsonify({
        "prediction": predicted_label,
        "confidence": round(confidence * 100, 2),
        "is_healthy": is_healthy,
        "top3": top3
    })


# ─── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)