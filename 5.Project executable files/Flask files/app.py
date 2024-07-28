from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import Xception, preprocess_input
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your pre-trained model (make sure to load it only once)
model = Xception(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/pred')
def pred():
    return render_template('details.html')

@app.route('/output', methods=['POST'])
def output():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the file content into memory
            file_content = file.read()
            image = Image.open(io.BytesIO(file_content))
            image = image.resize((299,299))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict using the pre-trained model
            preds = model.predict(img_array)
            print(preds[0])
            print("bhvkugcgcgh lgv")
            print(preds[0][0])
            result = 'Cataract was found, Kindly consult a doctor' if preds[0] > 0.5 else 'Congrats, Eye is Normal'
            return render_template("resu.html", result=result)
        else:
            return "No file uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
