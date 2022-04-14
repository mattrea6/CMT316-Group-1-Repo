import preporeseing
import train_model
import sys, os, shutil
import random

# main file to run all necessary routines

image_width = 256
image_height = 256

# this will split the data into train, test and val sets
preporeseing.preprocess()

# this trains and returns the model
model = train_model.train_model(image_width, image_height)
