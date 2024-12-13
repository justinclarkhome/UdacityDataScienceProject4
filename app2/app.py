from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os, sys
import plotly.express as px
import plotly.io as pio
from tensorflow.keras import models
from sklearn.datasets import load_files
from keras import utils as np_utils
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from keras.regularizers import l1, l2
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from PIL import ImageFile                            
from keras.callbacks import ModelCheckpoint  

sys.path.append('../')
from extract_bottleneck_features import extract_Resnet50


dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# load up the saved/trained bottleneck model
justinBottle_model = models.load_model('../saved_models/justinBottle.tf')

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def path_to_tensor(img_path):    
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img, verbose=False))


def predict_breed_from_image(img_path):
    """ Predict dog breed from image,

    Args:
        img_path (_type_): Path to image file.

    Returns:
        str: Predicted breed from image.
    """
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = justinBottle_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]    


def classify_image(img_path):
    # dog detected...
    is_dog = dog_detector(img_path)

    # person detected...
    is_human = face_detector(img_path)

    if is_dog or is_human:
        breed = predict_breed_from_image(img_path)
        if is_dog:
            answer = f'{img_path} contains a dog that looks like a: {breed}'
        else:
            answer = f'{img_path} contains a person who looks like a: {breed}'
    else:
        print(f'No dog or human was detected in this image: {img_path}')
        answer = -1
    return answer



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        breed = classify_image(file_path)
        print(breed)

        # Process the image (example: get image size)
        image = Image.open(file_path)
        image_info = f"Image size: {image.size[0]}x{image.size[1]}"
        
        # Create a simple Plotly figure as an example
        fig = px.imshow(image)
        plot_html = pio.to_html(fig, full_html=False)
        
        return render_template(
            'index.html', 
            filename=filename, 
            image_info=image_info, 
            plot_html=plot_html,
            )

if __name__ == '__main__':
    app.run(debug=True)
