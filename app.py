from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, url_for, send_from_directory
import numpy as np
import os

app = Flask(__name__)


model = load_model(r'C:\Users\P Alekhya\Desktop\Leader predicition\Quadleaders (1).h5', compile=False)


class_labels = ['Fumio Kishida', 'Joe Biden', 'Narendra Modi', 'Scott Morrison']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def upload_and_predict():
    if request.method == 'POST':
        
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/uploads', f.filename)
        f.save(upload_path)

        
        img = image.load_img(upload_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class_index]

        
        image_url = url_for('static', filename='uploads/' + f.filename)
        return render_template('result.html', prediction=predicted_label, image_url=image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/uploads'), filename)

if __name__ == "__main__":
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
