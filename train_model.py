import preporeseing
import os, shutil
import random
import pandas
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50

preporeseing.preprocess()
train_generator, test_generator, validation_generator = preporeseing.make_generators()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

base_model = ResNet50(input_shape=(256, 256, 3), include_top= False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

print("Creating model layers")
model=models.Sequential()
model.add(base_model)
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(1000,kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(200,activation='softmax'))
print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
print("Training model...")
history=model.fit(train_generator, steps_per_epoch=train_generator.samples/100, epochs=10, verbose= 1, validation_data= validation_generator)
