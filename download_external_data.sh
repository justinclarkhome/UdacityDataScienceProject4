#!/bin/bash

# download the human images
DIRECTORY="./lfw"
if [ ! -d "$DIRECTORY" ]; then
    mkdir -p "$DIRECTORY"
    echo "Directory created: $DIRECTORY"
else
    echo "Directory already exists: $DIRECTORY"
fi
curl --output ./lfw/lfw.zip https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
unzip ./lfw/lfw.zip

# download the dog images
DIRECTORY="./dogImages"
if [ ! -d "$DIRECTORY" ]; then
    mkdir -p "$DIRECTORY"
    echo "Directory created: $DIRECTORY"
else
    echo "Directory already exists: $DIRECTORY"
fi
curl --output ./dogImages/dogImages.zip https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip ./dogImages/dogImages.zip

# download bottleneck features
DIRECTORY="./bottleneck_features"
if [ ! -d "$DIRECTORY" ]; then
    mkdir -p "$DIRECTORY"
    echo "Directory created: $DIRECTORY"
else
    echo "Directory already exists: $DIRECTORY"
fi
curl --output ./bottleneck_features/DogResnet50Data.npz https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz
curl --output ./bottleneck_features/DogVGG16Data.npz https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz
