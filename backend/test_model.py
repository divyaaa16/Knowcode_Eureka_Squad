import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model("skin_disease_model.h5")
print("Model loaded successfully!")

# Class-to-disease mapping
class_to_disease = {
    0: "athelete-foot",
    1: "cellilitis",
    2: "chickenpox",
    3: "class_4",
    4: "cutaneous",
    5: "impetigo",
    6: "ringworm",
    7: "shingle"
}

# Function to preprocess and predict an image
def predict_image(image_path):
    try:
        # Load the image
        img = load_img(image_path, target_size=(224, 224))  # Resize image to model input size
        img_array = img_to_array(img) / 255.0              # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get class index
        confidence = np.max(prediction)                    # Get confidence level

        # Map class index to disease name
        disease_name = class_to_disease.get(predicted_class, "Unknown Disease")

        return disease_name, confidence
    except Exception as e:
        return f"Error processing image: {str(e)}", None


# Test on unseen images
test_images = [
    "test_images/img2.png",  # Replace with paths to your test images
]

for image_path in test_images:
    disease_name, confidence = predict_image(image_path)
    if confidence is not None:
        print(f"Image: {image_path}")
        print(f"Disease: {disease_name}, Confidence: {confidence:.2f}")
    else:
        print(disease_name)  # Display error message
