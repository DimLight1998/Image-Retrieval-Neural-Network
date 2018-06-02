import os
import json
import random

import IPython.display as display
import numpy as np
from PIL import Image as Image
from numpy.linalg import norm as norm
from tqdm import tqdm as tqdm
from keras.preprocessing import image as keras_image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


def cosine_distance(vec1, vec2):
    assert len(vec1) == len(vec2)
    return abs(np.inner(vec1, vec2) / (norm(vec1) * norm(vec2)))


if __name__ == '__main__':
    img_root = r"G:\Workspace\DS&Alg-Project1-Release\data\image"
    model = VGG16(weights='imagenet', include_top=False)

    # Train or load.
    use_cache = True
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

    # Test.
    top_k = 16
    test_images = os.listdir(os.path.join(img_root, 'test'))
    for test_image in test_images:
        image_path = os.path.join(img_root, 'test', test_image)
        image = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(image)
        x = np.expand_dims(x, 0)
        inputs = preprocess_input(x)
        feature = model.predict(inputs).reshape(7 * 7 * 512)
        answers = []
        for i in range(features.shape[0]):
            distance = cosine_distance(feature, features[i, :])
            answers.append((distance, train_images[i]))
        answers.sort(key=lambda x: x[0])
        answers = answers[:top_k]
        category_map = {pair[1].split('_')[0]: 0 for pair in answers}
        for pair in answers:
            category_map[pair[1].split('_')[0]] += 1
        category_predict = max(category_map.items(), key=lambda x: x[1])[0]
        if not category_predict == test_image.split('_')[0]:
            Image.open(os.path.join(img_root, 'test', test_image))
            os.system('pause')
            for pair in answers:
                Image.open(os.path.join(img_root, 'train', pair[1])).show()
            os.system('pause')
