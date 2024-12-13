from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import os, sys
import plotly.express as px
import plotly.io as pio
from tensorflow.keras import models
import numpy as np
from glob import glob
import cv2
from keras.preprocessing import image                  
from keras.applications.resnet50 import ResNet50, preprocess_input

sys.path.append('../')
from extract_bottleneck_features import extract_Resnet50


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# define folder to hold images the user uplaods
app.config['USER_IMAGE_UPLOAD_FOLDER'] = UPLOAD_FOLDER

# create that folder, if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ImageFile.LOAD_TRUNCATED_IMAGES = True                 

#################################################################################
##### Define items needed for model to make predictions on arbitrary images #####
#################################################################################

dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))] 

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def face_detector(img_path):
    """ returns "True" if face is detected in image stored at img_path.

    Args:
        img_path (str): Path to image file.

    Returns:
        bool: True if a face is detected, else False.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    """ returns "True" if dog is detected in image stored at img_path.

    Args:
        img_path (str): Path to image file.

    Returns:
        bool: True if a dog is detected, else False.
    """
    def ResNet50_predict_labels(img_path):
        ### returns prediction vector for image located at img_path ###
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img, verbose=False))

    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


def path_to_tensor(img_path):
    """ Load an image and convert to a 4D tensor.

    Args:
        img_path (str): Path to image file.

    Returns:
        np.array: 4d tensor array.
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def classify_image(img_path, model):
    """ Wrapper function to predict dog breed from an image using supplied model.
    The function will attempt to detect both dogs and people in the image. In either
    case, it will generate a predicted breed that the dog or person most resembles.

    Args:
        img_path (_tstrype_): Path to image file.
        model (object): Model used to make prediction.

    Returns:
        str: String containing a message including the dog breed detected.
    """
    
    def predict_breed(img_path, model):
        """ Predict dog breed from image.

        Args:
            img_path (_tstrype_): Path to image file.
            model (object): Model used to make prediction.

        Returns:
            str: Predicted breed from image.
        """
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]    
    
    is_dog = dog_detector(img_path) # dog detected
    is_human = face_detector(img_path) # person detected

    if is_dog or is_human:
        breed = predict_breed(img_path, model)
        if is_dog:
            answer = f'This image contains a dog that looks like: {breed}'
        else:
            answer = f'This image contains a person who looks like: {breed}'
    else:
        answer = 'No dog or human was detected in this image.'
    return answer


def load_justin_model(model_save_file='../saved_models/justin_model'):
    """ Load and return saved model for the image classification task.

    Args:
        model_save_file (str, optional): Path to stored model. Defaults to '../saved_models/justin_model'.

    Returns:
        object: An instance of the saved model.
    """
    bottleneck_features = np.load('../bottleneck_features/DogResNet50Data.npz')
    test = bottleneck_features['test']
    
    # load the on-disk model (this is defined/generated/saved in the ipynb file)
    try:
        m = models.load_model(model_save_file)
    except:
        print('Model could not be loaded. Run the dog_app.ipynb notebook to generate and save it.')

    # the loaded model doesn't process uploaded images properly, unless this line is run. I'm not sure why.
    p = [np.argmax(m.predict(np.expand_dims(feature, axis=0), verbose=0)) for feature in test]
    return m

# load the model (global scope so all functions can access it)
justinBottle_model = load_justin_model()

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
        file_path = os.path.join(app.config['USER_IMAGE_UPLOAD_FOLDER'], filename)

        file.save(file_path) # store the user's uploaded file locally
        
        # After the user uploads the file, predict the breed from it
        breed = classify_image(file_path, justinBottle_model)

        # Process the image (example: get image size)
        image = Image.open(file_path)
        
        # Display the image the user uploaded
        fig = px.imshow(image, title=breed)

        # hide the axis tick labels
        fig.update_layout(
            xaxis={'showticklabels': False},
            yaxis={'showticklabels': False},
        )
        plot_html = pio.to_html(fig, full_html=False)
        
        return render_template(
            'index.html', 
            filename=filename, 
            plot_html=plot_html,
            )


if __name__ == '__main__':
    app.run(debug=True)
