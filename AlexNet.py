#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.optimizers import Adam
from zipfile import ZipFile

# In[62]:


TrainingURL = r"E:\NN_project\Data\Train"

IMG_SIZE = 227


# In[63]:


# one hotencoding
def create_label(imagename):
    name = imagename.split('_')[0]
    if name == 'Basketball':
        return np.array([1, 0, 0, 0, 0, 0])
    elif name == 'Football':
        return np.array([0, 1, 0, 0, 0, 0])
    elif name == 'Rowing':
        return np.array([0, 0, 1, 0, 0, 0])
    elif name == 'Swimming':
        return np.array([0, 0, 0, 1, 0, 0])
    elif name == 'Tennis':
        return np.array([0, 0, 0, 0, 1, 0])
    elif name == 'Yoga':
        return np.array([0, 0, 0, 0, 0, 1])
    else:
        return np.array([0, 0, 0, 0, 0, 0])


def get_image_name(URL):
    return os.listdir(URL)


def create_data(URL_data):
    data = []

    for img in tqdm(get_image_name(URL_data)):
        path = os.path.join(URL_data, img)
        img_readed = cv2.imread(path)
        img_readed = cv2.resize(img_readed, (IMG_SIZE, IMG_SIZE))
        img_data = [np.array(img_readed), create_label(img)]
        data.append(img_data)

    type_data = URL_data.split('/')[-1]
    np.save(type_data + "_data.npy", data)
    return data


# In[64]:


# if data training is exist load file .npy
if (os.path.exists(r"E:\NN_project\Data\Train_data.npy")):
    train_data = np.load(r"E:\NN_project\Data\Train_data.npy", allow_pickle=True)
# else create file
else:
    train_data = create_data(TrainingURL)

# In[65]:


# Dataset

train_images = np.array([i[0] for i in train_data])
train_labels = np.array([i[1] for i in train_data])

train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels,
                                                                                    test_size=0.1, random_state=8)

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

print(type(train_ds))


# In[66]:


# Preprocessing
def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images to 227x227
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# Data/Input Pipeline
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
# print("Training data size:", train_ds_size)
# print("Validation data size:", validation_ds_size)

train_ds = (train_ds
            .map(process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=42, drop_remainder=True))
validation_ds = (validation_ds
                 .map(process_images)
                 .shuffle(buffer_size=validation_ds_size)
                 .batch(batch_size=13, drop_remainder=True))

# In[68]:


# Model Implementation
model = keras.models.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(227, 227, 3), ),

    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    keras.layers.Flatten(),

    keras.layers.Dense(4096),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(4096),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(1000),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(6),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('softmax'),
])

# In[69]:

#
# root_logdir = os.path.join(os.curdir, "logs\\fit\\")
# def get_run_logdir():
#     run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
#     return os.path.join(root_logdir, run_id)
# run_logdir = get_run_logdir()
# tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


# In[70]:


# load weights
# if os.path.exists(r'D:\1\7th term\NN\Project\CNN Project\AlexNet_weights.h5'):
#   model.load_weights(r'D:\1\7th term\NN\Project\CNN Project\AlexNet_weights.h5')
#   print("model weights loaded\n")
#

# compile
opt = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# In[ ]:


# model.summary()


# In[71]:
# callbacks=[tensorboard_cb]


if os.path.exists(r'E:\NN_project\AlexNet_Model.h5'):
    model = tf.keras.models.load_model(r'E:\NN_project\AlexNet_Model.h5')
else:
    model.fit(train_ds,
              epochs=40,
              validation_data=validation_ds,
              )
    model.save('ِAlexNet_Model.h5')
    model.save_weights('AlexNet_weights.h5')


# In[77]:


def predict(test_images, test_labels):
    images = []
    for image, label in zip(test_images, test_labels):
        images.append(process_images(image, label)[0])
    x_test = tf.data.Dataset.from_tensor_slices(images).batch(batch_size=16, drop_remainder=False)
    predictions = model.predict(x_test)
    labels = np.argmax(predictions, axis=1)
    return labels


# In[79]:


def get_csv(image_name, label):
    data = pd.DataFrame(data={'image_name': image_name, 'label': label})
    data.to_csv('AlexNet.csv', index=False)


# In[80]:


TestingURL = r"E:\NN_project\Data\Test"

# if data Testing is exist load file .npy
# if (os.path.exists(r'D:\1\7th term\NN\Project\CNN Project\Data\Test_data.npy')):
#     test_data = np.load(r'D:\1\7th term\NN\Project\CNN Project\Data\Test_data.npy',allow_pickle=True)
# #else create file
# else:
test_data = create_data(TestingURL)

test_images = np.array([i[0] for i in test_data])
test_labels = np.array([i[1] for i in test_data])

# In[81]:


test_label = predict(test_images, test_labels)
image_name = get_image_name(TestingURL)
get_csv(image_name, test_label)

