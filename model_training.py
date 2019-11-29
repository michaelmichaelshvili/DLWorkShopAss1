import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import os
import pandas as pd
import cv2
import numpy as np
import random


LABELS_PATH = r"dog-breed-identification\labels.csv"
IMAGES_PATH = r"dog-breed-identification\train_resize"
IMAGES_SIZE = 250
EPOCHS = 150
BATCH_SIZE = 32


def get_train_data(set_name):
    train_images = []
    train_labels = []
    labels_df = pd.read_csv(LABELS_PATH)
    labels_encoding = labels_df['breed'].str.get_dummies()
    for img in os.listdir(set_name):
        image_name = img[:-4]
        image_data = cv2.imread(os.path.join(set_name, image_name) + '.jpg')
        img_label = labels_df.loc[labels_df['id'] == image_name].index[0]
        train_images.append(np.array(image_data))
        train_labels.append(labels_encoding.iloc[img_label])
    return np.array(train_images), np.array(train_labels)

random.seed(1234)
img, lbl = get_train_data(IMAGES_PATH)
print('data has been loaded')

model = Sequential()
model.add(Conv2D(64, kernel_size = (3, 3), activation='relu', input_shape=(IMAGES_SIZE, IMAGES_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(Conv2D(386, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dense(120, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

print(model.summary())
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=8, verbose=1)
checkpointer = ModelCheckpoint(filepath='weights.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
model.fit(img, lbl, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.15, verbose=1, callbacks=[reduce_lr, checkpointer])

model.save('breedModel.h5')
