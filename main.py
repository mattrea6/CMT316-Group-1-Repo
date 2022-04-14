import preporeseing
import train_model
import os, shutil
import random
import pandas
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50

image_width = 256
image_height = 256

#preporeseing.preprocess()

model = train_model(image_width, image_height)
