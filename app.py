from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
from urllib.parse import quote
from flask import Flask

app = Flask(__name__)  # ✅ Define the Flask app

@app.route("/")
def home():
    return "Hello, Flask!"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


import signal

def handler(signum, frame):
    print(f"Received signal: {signum}")

# Load trained model
model = tf.keras.models.load_model("waste_classifier.h5")
CATEGORIES = ["Organic", "Recyclable", "Hazardous", "General"]

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    predicted_class = CATEGORIES[np.argmax(prediction)]

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
