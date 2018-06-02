import os

import numpy as np
import keras
from tqdm import tqdm as tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

import feature_extraction


def cosine_distance(vec1, vec2):
    assert len(vec1) == len(vec2)
    return abs(np.inner(vec1, vec2) / (norm(vec1) * norm(vec2)))


if __name__ == '__main__':
    img_root = r'G:\Workspace\DS&Alg-Project1-Release\data\image'
    images, features = feature_extraction.get_features(img_root)
    images_categories = list({img.split('_')[0] for img in images})
    category_map = {
        images_categories[i]: i for i in range(len(images_categories))}

    x_train = features
    labels = np.array(
        list(map(lambda x: category_map[x.split('_')[0]], images)))
    y_train = keras.utils.to_categorical(
        labels.reshape((x_train.shape[0], 1)), num_classes=len(images_categories))
    model = Sequential()

    #! Length hard coded.
    model.add(Dense(512, activation='relu', input_dim=7 * 7 * 512))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    #! Length hard coded.
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.001)
    model.compile(
        loss='categorical_crossentropy', optimizer=sgd, metrics=["acc"])
    model.fit(x_train, y_train, batch_size=128, epochs=16)

    # Test model.
    test_images = os.listdir(os.path.join(img_root, 'test'))
    labels = np.array(
        list(map(lambda x: category_map[x.split('_')[0]], test_images)))
    y_test = keras.utils.to_categorical(
        labels.reshape((len(labels), 1)), num_classes=len(images_categories))

    has_cache = True
    if not has_cache:
        # todo Use quicker method
        x_test = np.zeros((len(test_images), 7 * 7 * 512))
        for i in tqdm(range(len(test_images)), ascii=True):
            x_test[i, :] = feature_extraction.get_image_feature(
                os.path.join(img_root, 'test', test_images[i]))
        np.save('test_features.npy', x_test)
    else:
        x_test = np.load('test_features.npy')

    score = model.evaluate(x_test, y_test)
    print(score)
