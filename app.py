from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import io
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json, Sequential

app = Flask(__name__)
CORS(app)

# Load the pre-trained emotion model and its weights
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('model.weights.h5') # Note the updated filename here
    print("Model loaded successfully.")
    
    # Define emotion labels
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/predict_mood', methods=['POST'])
def predict_mood():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.json:
        return jsonify({'error': 'No image data found'}), 400

    image_data_base64 = request.json['image']

    try:
        image_bytes = base64.b64decode(image_data_base64)
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream)
        img_np = np.array(img)
        
        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Take the first detected face
            roi_gray = gray_image[y:y + h, x:x + w]
            
            # Resize for model input (48x48)
            resized_face = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
            
            # Normalize and reshape for prediction
            final_image = resized_face.astype('float32') / 255.0
            final_image = np.expand_dims(final_image, axis=0)
            final_image = np.expand_dims(final_image, axis=-1)
            
            # Predict emotion
            predictions = model.predict(final_image)
            dominant_emotion = emotion_labels[np.argmax(predictions)]
            
            print(f"Detected emotion: {dominant_emotion}")
            return jsonify({'mood': dominant_emotion, 'status': 'success'})
        else:
            return jsonify({'mood': 'unknown', 'status': 'face not detected or could not analyze'}), 200

    except Exception as e:
        print(f"Error during mood prediction: {e}")
        return jsonify({'error': 'An internal error occurred during mood detection'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)