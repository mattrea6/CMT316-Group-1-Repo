# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os, shutil
import pandas
from keras.preprocessing.image import ImageDataGenerator
import random

#Data set path (in the same directory as python files)
data_dir = './CUB_200_2011/CUB_200_2011'
image_dir = data_dir + '/' + 'images'
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')


#Function to create a new folder
def makedir(path_new):
    if not os.path.exists(path_new):
        os.mkdir(path_new)

# funciton to copy, move and resize image files
def preprocess():
    #Pandas loads the txt file to generate a two-dimensional array
    image_data = pandas.read_table('./CUB_200_2011/CUB_200_2011/images.txt', sep = ' ', names = ['id', 'names'])
    image_data = image_data.to_numpy()

    split_data = pandas.read_table('./CUB_200_2011/CUB_200_2011/train_test_split.txt', sep = ' ', names = ['id', 'train'])
    split_data = split_data.to_numpy()

    label_data = pandas.read_table('./CUB_200_2011/CUB_200_2011/image_class_labels.txt', sep = ' ', names = ['id', 'labels'])
    label_data = label_data.to_numpy()

    folders_data = pandas.read_table('./CUB_200_2011/CUB_200_2011/classes.txt', sep = ' ', names = ['id', 'classes'])
    folders_data = folders_data.to_numpy()

    print("Creating directories for train, test and validation")
    #Creating train, validation, test folders
    makedir(train_dir)
    makedir(validation_dir)
    makedir(test_dir)

    #List of test data set image names
    test_list = []

    print("Creating image directories")
    #Create separate folders for each class in the training set, test set, and validation set
    for i in range(len(folders_data)):
        makedir(train_dir + '/' + folders_data[i][1])
        makedir(validation_dir + '/' + folders_data[i][1])
        makedir(test_dir + '/' + folders_data[i][1])

    print("Copying files")
    #Split the test set and training set according to train_test_split.txt and copy the images to the appropriate folder
    data_length = len(split_data)
    for i in range(len(split_data)):
        print("Copying image {}/{}...".format(i, data_length), end="\r")
        if(split_data[i][1] == 1):
            src = os.path.join(image_dir, image_data[i][1])
            dst = os.path.join(train_dir, image_data[i][1])
            shutil.copyfile(src, dst)

        else:
            test_list.append(image_data[i][1])
            src = os.path.join(image_dir, image_data[i][1])
            dst = os.path.join(test_dir, image_data[i][1])
            shutil.copyfile(src, dst)

    #Half of the training set is randomly selected as the validation set
    validation_numb = int(round(len(test_list)*0.5 , 0))
    validation_list = random.sample(range(len(test_list)) , validation_numb)

    data_length = len(validation_list)
    print("Splitting test and validation sets")
    #Photos dividing test sets and validation sets
    for i, name in enumerate(test_list):
        if i in validation_list :
            print("Moving image {}/{}...".format(i, data_length), end="\r")
            src = os.path.join(test_dir, name)
            dst = os.path.join(validation_dir, name)
            shutil.move(src, dst)

# function to create and return image data generators for model.
def make_generators(width, height):
    print("Creating data generators")
    #Training set images for data Augmentation
    train_data = ImageDataGenerator(rescale=1./255,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range=0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip=True,)

    #All images of the test set and validation set are scaled by 1/255
    test_data = ImageDataGenerator(rescale=1./255)
    validation_data = ImageDataGenerator(rescale=1./255)

    #Resize the image to width*height
    train_generator = train_data.flow_from_directory(train_dir,target_size=(width, height),batch_size= 30)

    test_generator = test_data.flow_from_directory(test_dir,target_size=(width, height),batch_size = 30)

    validation_generator = validation_data.flow_from_directory(validation_dir,target_size=(width, height), batch_size=20, class_mode='categorical')

    return train_generator, test_generator, validation_generator
