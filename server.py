from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


app = Flask(__name__)


model = load_model("efficientnetb0_model_best_val_accuracy_epoch_117.keras")  


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html') 


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)


        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)

     
        prediction = model.predict(img_array)


        result = "Abnormal" if prediction[0][0] > 0.5 else "Normal"

        return render_template('result.html', result=result, img_path=filepath)


if __name__ == '__main__':
    app.run(debug=True)
