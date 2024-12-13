[This repo](https://github.com/justinclarkhome/UdacityDataScienceProject4?tab=readme-ov-file) stores code supporting my work for the capstone project of the Udacity Data Scientist nanodegree program.

The goal of the project is develop a convolutional neutral network (CNN) to detect the breed of a dog based on an image containing a dog. The code will take in an image, determine if a dog - or a person - is in the image, and then provide a prediction of the breed that the dog - or the person! - most resembles.

The Jupyter notebook contains the required code and Q&A for the project itself. When the notebook runs, it will build and save a CNN model (via bottleneck) that is then utilized a web app, where a user can upload an arbitrary image and obtain a breed prediction.

## Requirements
- The code runs in Python, and a **environment.yml** file is provided to allow the creation of an Anaconda environment containing all the libraries used by the author. Note the author used an Apple M3 laptop that requries the Apple channel for TensorFlow/Keras support.
- To create the environment, run "conda env create -f environment.yml".