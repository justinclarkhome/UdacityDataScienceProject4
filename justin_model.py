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

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


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


# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


def build_model(
    input_shape,
    n_target_classes,
    loss_function='categorical_crossentropy', # for mutually exclusive class labels (like we have)
    optimizer='adam',
    eval_metrics=['accuracy'],
    filters_layer1=32, # typically powers of 2, starting smaller and getting larger in later layers - this is for layer 1
    filters_other_layers=64, # ... and this is for any other layers added in
    conv_strides=(1, 1), # (1, 1) prob preferable, (2, 2) will reduce dimensions a bit more
    pool_size=(2, 2), # stick with (2, 2)
    kernel_size_layer1=(3, 3), # larger for larger details, smaller for finer details - this is for layer 1
    kernel_size_other_layers=(3, 3), # .... and this is for any other layers added in
):
        model = models.Sequential([
            # 1
            layers.Conv2D(
                filters=filters_layer1,
                kernel_size=kernel_size_layer1,
                strides=conv_strides,
                activation='relu', 
                padding='valid',
                input_shape=input_shape,
                ),
            layers.MaxPooling2D(pool_size=pool_size),
            
            # 2
            layers.Conv2D(
                filters=filters_other_layers,
                kernel_size=kernel_size_other_layers,
                strides=conv_strides,
                activation='relu',
                padding='valid',
                kernel_regularizer=l1(0.0002),
                # kernel_regularizer=l2(0.0002),
                ),
            layers.MaxPooling2D(pool_size=pool_size),

            layers.Flatten(),
            
            layers.Dense(n_target_classes, activation='relu'),
            layers.Dense(n_target_classes, activation='softmax')
        ])

        model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=eval_metrics,
                )
        
        return model