import numpy as np
# import cv2
# from keras.emotion_models import Sequential
from tensorflow.keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'archive/train'
validation_dir = 'archive/test'
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)


# Initialization Steps
trainGenerator = train_gen.flow_from_directory( # create a batch of tensor image data
    train_dir,
    target_size = (48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validationGenerator = val_gen.flow_from_directory(
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

# Training the model

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
train_gen = np.asarray(train_gen)
val_gen = np.asarray(val_gen)
emotion_model_info = emotion_model.fit(
        train_gen,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=val_gen,
        validation_steps=7178 // 64)

emotion_model.save_weights('model.h5')