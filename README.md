[This repo](https://github.com/justinclarkhome/UdacityDataScienceProject4?tab=readme-ov-file) stores code supporting my work for the capstone project of the Udacity Data Scientist nanodegree program.

# Problem statement/goal

The goal of the project is develop a convolutional neutral network (CNN) to detect the breed of a dog based on an image containing a dog. The code will take in an image, determine if a dog - or a person - is in the image, and then provide a prediction of the breed that the dog - or the person! - most resembles.

The Jupyter notebook contains the required code and Q&A for the project itself. When the notebook runs, it will build and save a CNN model (via bottleneck) that is then utilized a web app, where a user can upload an arbitrary image and obtain a breed prediction. There is also an HTML rendering of the executed notebook, for reference.

## Requirements
- The code runs in Python, and a **environment.yml** file is provided to allow the creation of an Anaconda environment containing all the libraries used by the author. 
	- **Note**: the author used an Apple M3 laptop that requries the Apple channel for TensorFlow/Keras support.
	- If the user is not using an Apple silicon platform, remove the 'Apple' channel at the top of the YAML file and remove the 'tensorflow-macos' and 'tensorflow-metal' lines in the pip section at the bottom (and install TensorFlow/Keras as necessary for your platform).
	- If not using conda, you will need to at least install numpy, tensorflow, scikit-learn, plotly,flask, matplotlib and opencv (which may be via opencv-python-headless or similar, depending on your platform).
	- **Note**: I uploaded this code to the Udacity virtual Jupyer virtual instance, but got repeated kernel crashes while pre-processing the image data. This is why I moved the code to Github.
- To create the environment, run "conda env create -f environment.yml".
- Some data files are required, but too large to store in Github.
	- If you are on MacOS or Linux, you can run the download_externel_data.sh script to download the required external data, e.g. "sh ./download_external_data.sh" from the command line. This will take several minutes!
	- Otherwise, you can do this manually:
		- Create a directory in the root of the repo called 'dogImages', download [this](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and extract it in that directory.
		- Create a directory in the root of the repo called 'lfw', downlaod [this](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) and extract it in that directory.
		- Create a directory in the root of the repo called 'bottleneck_features', and download [this](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) and [this](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) into that directory.
- Run the dog_app.ipnb file in its entirety, which will save a model in the 'saved_models' directory, which the web app will load.

## Running the web app
- From the app directory, run "python app.py" to launch the web app.
- Allow a few seconds for the model to load, and then click the URL the terminal shows to access the webapp.
- From there, you can upload an arbitrary image: several are located in the 'downloaded_test_images' directory of this repo (4 of dogs, 2 of humans, and one with neither).

# Acknowledgements
The author's experience with CNNs was limited prior to working on this project, and several sources were VERY helpful in working through the various sections of code:
- [DataCamp](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns)
- [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/?utm_source=blog&utm_source=learn-image-classification-cnn-convolutional-neural-networks-5-datasets)
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials/images/classification)
- [GeeksForGeeks](https://www.geeksforgeeks.org/python-image-classification-using-keras/)
- [TowardsDataScience](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)
- [Microsoft's Copilot AI](https://copilot.microsoft.com/): While I avoided having AI tools generate much code for me, Copilot was  helpful in making some boilerplate suggestions that reduced the amount of time I spent figuring out how to build the web app in the way I desired (e.g. the interactions between Python and Flask/Bootstrap/Plotly). 

