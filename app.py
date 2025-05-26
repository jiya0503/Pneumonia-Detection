from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.saving import register_keras_serializable
from keras.saving import register_keras_serializable
import numpy as np
import tensorflow as tf
import os

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@register_keras_serializable(package="custom")
def focal_loss(alpha=0.25, gamma=2.0):
    @register_keras_serializable(package="custom")
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss = -y_true * alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss -= (1 - y_true) * (1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return loss

# Load your model
MODEL_PATH = os.path.join('model', 'pneumonia_densenet.keras')
model = load_model(MODEL_PATH, custom_objects={'focal_loss': focal_loss(alpha=0.25, gamma=2.0)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="⚠ No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="⚠ No file selected")

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = [model.predict(img_array)[0][0] for _ in range(10)]
        final_prediction = np.mean(predictions)

        label = "🦠 Pneumonia Detected" if final_prediction > 0.5 else "✅ Normal"
        os.remove(filepath)

        return render_template('index.html', prediction=label)

    return render_template('index.html', prediction="⚠ Something went wrong")

if _name_ == '_main_':
    app.run(debug=True)