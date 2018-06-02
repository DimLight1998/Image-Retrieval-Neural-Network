import os
import json

import numpy as np
from tqdm import tqdm as tqdm
from keras.preprocessing import image as keras_image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


def get_features(img_root, use_cache=True):
    model = VGG16(weights='imagenet', include_top=False)

    # Train or load.
    if not use_cache:
        train_images = os.listdir(os.path.join(img_root, 'train'))
        with open('train_images.json', 'w+') as f:
            json.dump(train_images, f)

        inputs = np.zeros((len(train_images), 224, 224, 3))
        for i, image_name in tqdm(list(enumerate(train_images)), ascii=True):
            image = keras_image.load_img(os.path.join(
                img_root, 'train', image_name), target_size=(224, 224))
            x = keras_image.img_to_array(image)
            inputs[i, :, :, :] = x

        inputs = preprocess_input(inputs)

        features = np.zeros((len(train_images), 7 * 7 * 512))
        for i in tqdm(range(len(train_images)), ascii=True):
            single_input = inputs[i:i+1, :, :, :]
            feature = model.predict(single_input)
            features[i, :] = feature.reshape((7 * 7 * 512))
        np.save('features.npy', features)
    else:
        with open('train_images.json') as f:
            train_images = json.load(f)
        features = np.load('features.npy')

    return train_images, features

def get_image_feature(imagepath):
    image = keras_image.load_img(imagepath, target_size=(224, 224))
    x = keras_image.img_to_array(image)
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)
    feature = VGG16(weights='imagenet', include_top=False).predict(x)
    feature = feature.reshape((7 * 7 * 512))
    return feature
