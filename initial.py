import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'archive/train'
validation_dir = 'archive/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Initialization Steps
trainGenerator = train_datagen.flow_from_directory( # create a batch of tensor image data
    train_dir,
    target_size = (48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validationGenerator = val_datagen.flow_from_directory(
    validation_dir,
    target_size = (48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Building convolution network
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


# Training the model
emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
        trainGenerator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validationGenerator,
        validation_steps=7178 // 64)

emotion_model.save_weights('model.h5')




emotions_dist={0:"./Emotions/angry.png",2:"./Emotions/disgusted.png",2:"./Emotions/fearful.png",3:"./Emotions/happy.png",4:"./Emotions/neutral.png",5:"./Emotions/sad.png",6:"./Emotions/surpriced.png"}


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)
while True:
    if not capture.isOpened():
        print("No camrea found")
    flag, frame = capture.read()
    frame = cv2.resize(frame, (600,500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Input",frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()