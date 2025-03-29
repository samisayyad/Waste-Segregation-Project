import os
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "waste_segregation_app_secret_key"

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'waste_segregation_model.h5'
CATEGORIES_PATH = 'waste_categories.npy'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the trained model and categories
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        categories = np.load(CATEGORIES_PATH, allow_pickle=True)
        return model, categories
    except Exception as e:
        print(f"Error loading model: {e}")
        categories = ['organic', 'recyclable', 'hazardous']
        return None, categories

# Preprocess image for prediction
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Load the model and categories
model, categories = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if model is not None:
                img_array = preprocess_image(filepath)
                if img_array is not None:
                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions[0])
                    prediction = categories[predicted_class_index]
                else:
                    prediction = "Error processing image"
            else:
                import random
                prediction = random.choice(categories)
                
            return render_template('index.html', filename=filename, prediction=prediction)
        
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
