import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os

app = Flask(__name__)

path_simple = 'modele/simple_model.h5'
path_advanced = 'modele/advanced_model.h5'

if not os.path.exists(path_simple) or not os.path.exists(path_advanced):
    print("BŁĄD: Nie znaleziono modeli! Najpierw uruchom 'train_models.py'.")
    exit()

MODEL_SIMPLE = load_model(path_simple)
MODEL_ADVANCED = load_model(path_advanced)

CLASS_NAMES = ['T-shirt/top', 'Spodnie', 'Sweter', 'Sukienka', 'Płaszcz',
               'Sandał', 'Koszula', 'Trampek', 'Torebka', 'But']

def prepare_image(img_file):
    img = Image.open(img_file).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array[img_array < 0.3] = 0.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_simple = ""
    prediction_advanced = ""
    msg = ""
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', msg="Nie przesłano pliku")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', msg="Nie wybrano pliku")

        if file:
            try:
                img_array = prepare_image(file)
                
                input_simple = img_array.reshape(1, 28, 28)
                pred_s = MODEL_SIMPLE.predict(input_simple)
                class_s = CLASS_NAMES[np.argmax(pred_s)]
                conf_s = np.max(pred_s) * 100
                
                input_adv = img_array.reshape(1, 28, 28, 1)
                pred_a = MODEL_ADVANCED.predict(input_adv)
                class_a = CLASS_NAMES[np.argmax(pred_a)]
                conf_a = np.max(pred_a) * 100
                
                prediction_simple = f"{class_s} ({conf_s:.2f}%)"
                prediction_advanced = f"{class_a} ({conf_a:.2f}%)"
            except Exception as e:
                msg = f"Wystąpił błąd: {e}"

    return render_template('index.html', 
                           res_simple=prediction_simple, 
                           res_advanced=prediction_advanced,
                           msg=msg)

if __name__ == '__main__':
    app.run(debug=True)