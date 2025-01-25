from flask import Flask, request, jsonify, render_template, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model("C:\\Users\\karpe\\OneDrive\\Desktop\\skin\\skin_disease_model.h5")

# Class-to-disease mapping
class_to_disease = {
    0: "Acne",
    1: "Eczema",
    2: "Psoriasis",
    3: "Rosacea",
    4: "Melanoma",
    5: "Urticaria",
    6: "Vitiligo",
    7: "Lichen Planus"
}

@app.route('/')
def home():
    return render_template('index.html')  # This will render the full template

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Create test_images directory if it doesn't exist
        os.makedirs('test_images', exist_ok=True)
        
        # Save the uploaded file to test_images folder
        filepath = os.path.join('test_images', file.filename)
        file.save(filepath)
        
        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class] * 100)
        
        # Get disease name
        disease_name = class_to_disease.get(predicted_class, "Unknown Disease")
        
        return jsonify({
            'disease': disease_name,
            'confidence': confidence
        })
                             
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    filename = image.filename
    filepath = os.path.join("uploaded_images", filename)
    os.makedirs("uploaded_images", exist_ok=True)
    image.save(filepath)

    try:
        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        # Map class to disease name
        disease_name = class_to_disease.get(predicted_class, "Unknown Disease")

        # Debugging
        print(f"Predictions Array: {predictions}")
        print(f"Predicted Class Index: {predicted_class}")
        print(f"Disease Name Retrieved: {disease_name}")
        print(f"Confidence Level: {confidence}")

        return jsonify({
            "message": "Prediction successful",
            "disease": disease_name,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

