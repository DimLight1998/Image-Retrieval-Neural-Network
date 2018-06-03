import os
import random
import yaml
import numpy as np
import keras
from tqdm import tqdm as tqdm
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image as keras_image
from keras.optimizers import SGD


# Load configuration.
with open('config.yaml') as f:
    config = yaml.load(f)

# Load data.
pictures = os.listdir(config['image_path'])
random.shuffle(pictures)
print('Constructing features')
features = np.zeros((len(pictures), 224, 224, 3))
for i, image_name in tqdm(list(enumerate(pictures)), ascii=True):
    image = keras_image.load_img(os.path.join(
        config['image_path'], image_name), target_size=(224, 224))
    features[i, :, :, :] = keras_image.img_to_array(image)
print('Constructing labels')
categories = list(set(map(lambda x: x.split('_')[0], pictures)))
category_map = {categories[i]: i for i in range(len(categories))}
labels = np.array(
    list(map(lambda x: category_map[x.split('_')[0]], pictures))).reshape((-1, 1))
labels = keras.utils.to_categorical(labels, num_classes=len(categories))


# Construct and train model.
if config['use_pretrained_vgg16']:
    model = load_model('./vgg16.hdf5')
else:
    inputs_layer = Input(shape=(224, 224, 3))
    vgg_model = VGG16(weights='imagenet', include_top=False)
    vgg_model.trainable = False
    conv_result = vgg_model(inputs_layer)
    flattend_layer = Flatten()(conv_result)
    dense1 = Dense(512, activation='relu')(flattend_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(512, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    dense3 = Dense(10, activation='softmax')(dropout2)
    model = Model(inputs=inputs_layer, outputs=dense3)

sgd = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
history = model.fit(features, labels, batch_size=4, epochs=16, validation_split=0.1)
print(history.history)
model.save('vgg16.h5')