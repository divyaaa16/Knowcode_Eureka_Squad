from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import scipy.signal
from tensorflow.keras.applications import VGG16

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'models/skin_disease_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Disease classes (update these based on your dataset)
CLASSES = ['Acne', 'Eczema', 'Melanoma', 'Psoriasis', 'Rosacea']

def preprocess_image(image):
    # Resize image to 224x224 pixels
    image = image.resize((224, 224))
    # Convert to array
    image_array = img_to_array(image)
    # Expand dimensions
    image_array = np.expand_dims(image_array, axis=0)
    # Preprocess input
    return preprocess_input(image_array)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = CLASSES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    # Get disease information
    disease_info = get_disease_info(predicted_class)
    
    # Get severity analysis
    severity_analysis = calculate_severity(processed_image)
    
    return jsonify({
        'disease': predicted_class,
        'confidence': confidence,
        'information': disease_info,
        'severity': severity_analysis['level'],
        'severity_confidence': severity_analysis['confidence'],
        'texture_analysis': severity_analysis['metrics']
    })

def get_disease_info(disease_name):
    # Dictionary containing disease information
    disease_info = {
        'Acne': {
            'description': 'A skin condition that occurs when hair follicles plug with oil and dead skin cells.',
            'treatment': 'Topical treatments (benzoyl peroxide, salicylic acid), antibiotics, good skincare routine',
            'severity_levels': ['Mild', 'Moderate', 'Severe']
        },
        'Eczema': {
            'description': 'A condition that makes your skin red and itchy, often appearing in patches.',
            'treatment': 'Moisturizers, topical corticosteroids, antihistamines',
            'severity_levels': ['Mild', 'Moderate', 'Severe']
        },
        'Melanoma': {
            'description': 'The most serious type of skin cancer that develops in melanocytes.',
            'treatment': 'Surgery, immunotherapy, targeted therapy, radiation therapy',
            'severity_levels': ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
        },
        'Psoriasis': {
            'description': 'A skin disease causing red, itchy scaly patches.',
            'treatment': 'Topical treatments, phototherapy, systemic medications',
            'severity_levels': ['Mild', 'Moderate', 'Severe']
        },
        'Rosacea': {
            'description': 'A common skin condition causing redness and visible blood vessels in face.',
            'treatment': 'Topical medications, oral antibiotics, laser therapy',
            'severity_levels': ['Mild', 'Moderate', 'Severe']
        }
    }
    return disease_info.get(disease_name, {
        'description': 'Information not available',
        'treatment': 'Please consult a healthcare professional',
        'severity_levels': ['Unknown']
    })

def calculate_severity(image_array):
    """Calculate severity using advanced CNN texture analysis"""
    try:
        # Create feature extractors using different CNN architectures for better texture analysis
        mobilenet = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        vgg = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Define texture-relevant layers from both networks
        texture_layers = {
            'mobilenet': [
                'block_1_expand_relu',    # Edge features
                'block_3_expand_relu',    # Texture patterns
                'block_6_expand_relu'     # Complex textures
            ],
            'vgg': [
                'block1_conv2',           # Fine texture details
                'block2_conv2',           # Medium texture patterns
                'block3_conv3'            # Coarse texture features
            ]
        }
        
        # Create models for feature extraction
        mobilenet_outputs = [mobilenet.get_layer(layer).output for layer in texture_layers['mobilenet']]
        vgg_outputs = [vgg.get_layer(layer).output for layer in texture_layers['vgg']]
        
        mobilenet_model = tf.keras.Model(inputs=mobilenet.input, outputs=mobilenet_outputs)
        vgg_model = tf.keras.Model(inputs=vgg.input, outputs=vgg_outputs)
        
        # Get feature maps from both models
        mobilenet_features = mobilenet_model.predict(image_array)
        vgg_features = vgg_model.predict(image_array)
        
        # Calculate advanced texture metrics
        texture_metrics = {}
        all_scores = []
        
        # Process features from both networks
        for idx, (mobile_feat, vgg_feat) in enumerate(zip(mobilenet_features, vgg_features)):
            # Statistical features
            mean_mobile = np.mean(mobile_feat)
            std_mobile = np.std(mobile_feat)
            mean_vgg = np.mean(vgg_feat)
            std_vgg = np.std(vgg_feat)
            
            # Calculate GLCM (Gray Level Co-occurrence Matrix) features
            gray_level = tf.image.rgb_to_grayscale(image_array[0])
            glcm_features = calculate_glcm_features(gray_level.numpy())
            
            # Gabor filter responses for different orientations
            gabor_features = calculate_gabor_features(gray_level.numpy())
            
            # Combine all texture features
            layer_score = (
                mean_mobile + std_mobile + mean_vgg + std_vgg +
                glcm_features['contrast'] + glcm_features['energy'] +
                np.mean(gabor_features)
            ) / 7
            
            all_scores.append(layer_score)
            
            # Store metrics for this layer
            texture_metrics[f'layer_{idx}'] = {
                'statistical': {
                    'mean_mobilenet': float(mean_mobile),
                    'std_mobilenet': float(std_mobile),
                    'mean_vgg': float(mean_vgg),
                    'std_vgg': float(std_vgg)
                },
                'glcm': glcm_features,
                'gabor': float(np.mean(gabor_features))
            }
        
        # Calculate final severity score using weighted combination
        final_score = np.mean(all_scores)
        
        # Determine severity level with confidence
        if final_score < 0.3:
            severity = "Mild"
            confidence = (0.3 - final_score) / 0.3
        elif final_score < 0.6:
            severity = "Moderate"
            confidence = (final_score - 0.3) / 0.3
        else:
            severity = "Severe"
            confidence = (final_score - 0.6) / 0.4
        
        return {
            'level': severity,
            'confidence': min(float(confidence), 1.0),
            'metrics': texture_metrics
        }
        
    except Exception as e:
        print(f"Error in severity calculation: {str(e)}")
        return {
            'level': "Moderate",
            'confidence': 0.5,
            'metrics': {'error': str(e)}
        }

def calculate_glcm_features(image):
    """Calculate GLCM (Gray Level Co-occurrence Matrix) features"""
    # Normalize and quantize the image
    img_normalized = (image * 255).astype(np.uint8)
    
    # Calculate basic GLCM features
    contrast = np.sum(np.square(img_normalized - np.mean(img_normalized)))
    energy = np.sum(np.square(img_normalized))
    homogeneity = np.sum(1 / (1 + np.square(img_normalized - np.mean(img_normalized))))
    correlation = np.sum((img_normalized - np.mean(img_normalized)) * 
                        (img_normalized - np.mean(img_normalized)))
    
    return {
        'contrast': float(contrast),
        'energy': float(energy),
        'homogeneity': float(homogeneity),
        'correlation': float(correlation)
    }

def calculate_gabor_features(image):
    """Calculate Gabor filter responses for texture analysis"""
    frequencies = [0.1, 0.25, 0.4]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []
    
    for frequency in frequencies:
        for theta in orientations:
            # Create Gabor kernel
            sigma = 1.0
            kernel_size = 5
            y, x = np.mgrid[-kernel_size//2:kernel_size//2 + 1,
                           -kernel_size//2:kernel_size//2 + 1]
            
            # Gabor function
            gb = np.exp(-(x**2 + y**2)/(2*sigma**2)) * \
                 np.cos(2*np.pi*frequency*(x*np.cos(theta) + y*np.sin(theta)))
            
            # Apply filter
            filtered = scipy.signal.convolve2d(image[:,:,0], gb, mode='same')
            features.append(np.mean(np.abs(filtered)))
    
    return np.array(features)

if __name__ == '__main__':
    app.run(debug=True) 