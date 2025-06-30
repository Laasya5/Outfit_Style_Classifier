import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Change this to your actual names if used in your UI
student_names = ['Laasya Reddy', 'Student 2', 'Student 3']

# Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), 'static', 'models', 'mobilenetv2_outfit_classifier.h5')
model = load_model(MODEL_PATH)

# Define constants
IMAGE_SIZE = (224, 224)  # or (120, 90) if that's what your model expects
CLASS_LABELS = sorted([
    'T-Shirt/Top', 'Shirt', 'Blouse', 'Sweater', 'Jacket/Coat', 'Dress',
    'Skirt', 'Trousers/Jeans', 'Shorts', 'Shoes', 'Sandals', 'Boots',
    'Heels', 'Socks', 'Hat/Cap', 'Glasses/Sunglasses', 'Bag/Purse',
    'Scarf', 'Watch', 'Belt'
])  # Ensure this matches the label encoding used in training

@app.route("/", methods=["GET"], strict_slashes=False)
def home():
    return render_template("index.html", students=student_names)

@app.route("/classify", methods=["POST"], strict_slashes=False)
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    uploaded_file = request.files["image"]

    try:
        # Load and preprocess the image
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        accuracy = float(np.max(predictions[0])) * 100

        predicted_label = CLASS_LABELS[predicted_class]

        return jsonify({
            "class": predicted_label,
            "accuracy": f"{accuracy:.2f}%",
            "loss": f"{loss:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
