import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import cv2
import tempfile
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('wall_quality_model.h5')

def enhance_texture_single(image_path):
    
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)
    
    smoothed_image = cv2.bilateralFilter(equalized_image, d=9, sigmaColor=75, sigmaSpace=75)
    
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened_image = cv2.filter2D(smoothed_image, -1, sharpening_kernel)
    
    edges = cv2.Canny(sharpened_image, 50, 150)
    
    combined_image = cv2.addWeighted(sharpened_image, 0.8, edges, 0.2, 0)
    
    return combined_image

def predict(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0

    prediction = model.predict(image_array)
    print(f"Prediction: {prediction}")
    if prediction < 0.5:
        print("The image is classified as bad")
        message = "The image is classified as bad"
        return message
    else:
        message = "The image is classified as good"
        print("The image is classified as good")
        return message

@app.route('/predict', methods=['POST'])
def home():
    image_file = request.files['image']
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image_path = temp_image.name
    image_file.save(image_path)
    
    combined_image = enhance_texture_single(image_path)
    
    temp_enhanced_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    enhanced_image_path = temp_enhanced_image.name
    cv2.imwrite(enhanced_image_path, combined_image)

    message = predict(enhanced_image_path)
    
    os.remove(image_path)
    os.remove(enhanced_image_path)
    
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
