import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Upload folder for prediction images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "plant_disease_model.h5")
model = load_model(MODEL_PATH)

# Disease classes and prevention tips
disease_classes = {
    0: "Apple Scab", 1: "Apple Black Rot", 2: "Apple Cedar Rust", 3: "Apple Healthy",
    4: "Blueberry Healthy", 5: "Cherry Powdery Mildew", 6: "Cherry Healthy",
    7: "Corn Gray Leaf Spot", 8: "Corn Common Rust", 9: "Corn Northern Leaf Blight", 10: "Corn Healthy",
    11: "Grape Black Rot", 12: "Grape Esca (Black Measles)", 13: "Grape Leaf Blight", 14: "Grape Healthy",
    15: "Orange Huanglongbing (Citrus Greening)", 16: "Peach Bacterial Spot", 17: "Peach Healthy",
    18: "Pepper Bell Bacterial Spot", 19: "Pepper Bell Healthy", 20: "Potato Early Blight", 21: "Potato Late Blight",
    22: "Potato Healthy", 23: "Raspberry Healthy", 24: "Soybean Healthy", 25: "Squash Powdery Mildew",
    26: "Strawberry Leaf Scorch", 27: "Strawberry Healthy", 28: "Tomato Bacterial Spot", 29: "Tomato Early Blight",
    30: "Tomato Late Blight", 31: "Tomato Leaf Mold", 32: "Tomato Septoria Leaf Spot", 33: "Tomato Spider Mite",
    34: "Tomato Target Spot", 35: "Tomato Mosaic Virus", 36: "Tomato Yellow Leaf Curl Virus", 37: "Tomato Healthy"
}

preventions = {
    0: "Use resistant varieties and practice crop rotation.", 1: "Remove infected fruit and apply appropriate fungicides.",
    2: "Remove infected leaves and apply fungicide.", 3: "No action needed. The plant is healthy.",
    4: "No action needed. The plant is healthy.", 5: "Apply sulfur-based fungicides and prune infected parts.",
    6: "No action needed. The plant is healthy.", 7: "Use fungicide sprays and plant resistant varieties.",
    8: "Use fungicides and rotate crops.", 9: "Use resistant hybrids and fungicides.",
    10: "No action needed. The plant is healthy.", 11: "Prune infected parts and apply fungicides.",
    12: "Use proper pruning and fungicides.", 13: "Remove infected leaves and use fungicide.",
    14: "No action needed. The plant is healthy.", 15: "Remove infected trees and control psyllid vector.",
    16: "Remove infected leaves and use copper-based sprays.", 17: "No action needed. The plant is healthy.",
    18: "Apply copper-based bactericides.", 19: "No action needed. The plant is healthy.",
    20: "Remove infected leaves and apply fungicide.", 21: "Use certified seeds and fungicides.",
    22: "No action needed. The plant is healthy.", 23: "No action needed. The plant is healthy.",
    24: "No action needed. The plant is healthy.", 25: "Use fungicide sprays and proper spacing.",
    26: "Remove infected leaves and apply fungicides.", 27: "No action needed. The plant is healthy.",
    28: "Apply copper-based bactericides.", 29: "Use crop rotation and fungicides.",
    30: "Apply fungicides and remove infected plants.", 31: "Improve air circulation and apply fungicides.",
    32: "Remove infected leaves and apply fungicides.", 33: "Use insecticidal soaps and remove webs.",
    34: "Use proper spacing and fungicides.", 35: "Remove infected plants and control aphids.",
    36: "Use insecticides and resistant varieties.", 37: "No action needed. The plant is healthy."
}

# Crop Data
crop_data = {
    ('Kharif', 'High'): [
        {'name': 'Rice', 'image': 'static/images/rice.jpg', 'growth_time': '4-5 months'},
        {'name': 'Sugarcane', 'image': 'static/images/sugarcane.jpg', 'growth_time': '9-12 months'},
        {'name': 'Jute', 'image': 'static/images/jute.jpg', 'growth_time': '3-4 months'},
        {'name': 'Banana', 'image': 'static/images/banana.jpg', 'growth_time': '9-10 months'},
        {'name': 'Cotton', 'image': 'static/images/cotton.jpg', 'growth_time': '5-6 months'}
    ],
    ('Kharif', 'Moderate'): [
        {'name': 'Maize', 'image': 'static/images/maize.jpg', 'growth_time': '3 months'},
        {'name': 'Soybean', 'image': 'static/images/soybean.jpg', 'growth_time': '3-4 months'},
        {'name': 'Tur (Pigeon Pea)', 'image': 'static/images/tur.jpg', 'growth_time': '5-7 months'},
        {'name': 'Groundnut', 'image': 'static/images/groundnut.jpg', 'growth_time': '3-4 months'},
        {'name': 'Okra (Lady Finger)', 'image': 'static/images/okra.jpg', 'growth_time': '2 months'}
    ],
    ('Kharif', 'Low'): [
        {'name': 'Sesame', 'image': 'static/images/sesame.jpg', 'growth_time': '2.5 months'},
        {'name': 'Sunflower', 'image': 'static/images/sunflower.jpg', 'growth_time': '3 months'}
    ],
    ('Rabi', 'High'): [
        {'name': 'Sugar Beet', 'image': 'static/images/sugar_beet.jpg', 'growth_time': '3-4 months'},
        {'name': 'Barley', 'image': 'static/images/barley.jpg', 'growth_time': '5-6 months'}
    ],
    ('Rabi', 'Moderate'): [
        {'name': 'Pea', 'image': 'static/images/pea.jpg', 'growth_time': '2-3 months'},
        {'name': 'Carrot', 'image': 'static/images/carrot.jpg', 'growth_time': '2.5-3 months'},
        {'name': 'Cauliflower', 'image': 'static/images/cauliflower.jpg', 'growth_time': '3-4 months'},
        {'name': 'Cabbage', 'image': 'static/images/cabbage.jpg', 'growth_time': '3-4 months'}
    ],
    ('Rabi', 'Low'): [
        {'name': 'Wheat', 'image': 'static/images/wheat.jpg', 'growth_time': '5-6 months'},
        {'name': 'Mustard', 'image': 'static/images/mustard.jpg', 'growth_time': '4-5 months'},
        {'name': 'Gram (Chickpea)', 'image': 'static/images/chickpea.jpg', 'growth_time': '3.5-4 months'},
        {'name': 'Garlic', 'image': 'static/images/garlic.jpg', 'growth_time': '4-5 months'}
    ],
    ('Zaid', 'High'): [
        {'name': 'Pumpkin', 'image': 'static/images/pumpkin.jpg', 'growth_time': '2.5-3 months'},
        {'name': 'Musk Melon', 'image': 'static/images/muskmelon.jpg', 'growth_time': '3 months'}
    ],
    ('Zaid', 'Moderate'): [
        {'name': 'Watermelon', 'image': 'static/images/watermelon.jpg', 'growth_time': '3 months'},
        {'name': 'Cucumber', 'image': 'static/images/cucumber.jpg', 'growth_time': '1.5-2 months'},
        {'name': 'Bottle Gourd', 'image': 'static/images/bottle_gourd.jpg', 'growth_time': '2-2.5 months'}
    ],
    ('Zaid', 'Low'): [
        {'name': 'Zinnia (Flower)', 'image': 'static/images/zinnia.jpg', 'growth_time': '2-2.5 months'},
        {'name': 'Marigold (Flower)', 'image': 'static/images/marigold.jpg', 'growth_time': '2.5-3 months'}
    ]
}


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop-disease-predictor', methods=['GET'])
def crop_disease_predictor():
    return render_template('crop_disease_predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('crop_disease_predictor'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('crop_disease_predictor'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_disease = disease_classes.get(predicted_class_index, "Unknown Disease")
    prevention_tip = preventions.get(predicted_class_index, "No prevention tip available.")

    return render_template(
        'crop_disease_predictor.html',
        prediction=predicted_disease,
        prevention=prevention_tip,
        image_path=f"uploads/{filename}"
    )

@app.route('/crop-suggestor', methods=['GET', 'POST'])
def crop_suggestor():
    suggested_crops = []
    if request.method == 'POST':
        season = request.form.get('season')
        water = request.form.get('water')
        
        suggested_crops = crop_data.get((season, water), [])
    return render_template('crop_suggestor.html',crops=suggested_crops)
@app.route('/videos', methods=['GET'])
def videos():
    return render_template('videos.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)