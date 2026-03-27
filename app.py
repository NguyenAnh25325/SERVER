from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# =========================
# 🔥 TỐI ƯU RAM TENSORFLOW
# =========================
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================
# 🚀 INIT APP
# =========================
app = Flask(__name__)

# =========================
# 📦 LOAD MODEL (CHỈ 1 LẦN)
# =========================
print("🔄 Loading model...")

model = tf.keras.models.load_model(
    "model/plant_model_final.keras",
    compile=False
)

print("✅ Model loaded successfully")


# =========================
# 🧠 CLASS LABEL (SỬA THEO MODEL CỦA BẠN)
# =========================
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___healthy"
]


# =========================
# 🔍 PREPROCESS IMAGE
# =========================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  # chỉnh theo model bạn train
    img = img.convert("RGB")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================
# ❤️ HOME ROUTE
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OK",
        "message": "Plant Disease API is running"
    })


# =========================
# 🔥 PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image_bytes = file.read()

        # preprocess
        img = preprocess_image(image_bytes)

        # predict
        preds = model.predict(img)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        result = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else "Unknown"

        return jsonify({
            "class_index": class_index,
            "confidence": round(confidence, 2),
            "result": result,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# 🚀 RUN (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
