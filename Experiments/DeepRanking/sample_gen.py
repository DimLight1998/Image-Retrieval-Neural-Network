import json
import os
import random
import time

import numpy as np
from keras.preprocessing import image as keras_image
from Experiments.DeepRanking.similar_img_gen import similar_image as similar_image

from matplotlib import pyplot as plt


def generate_triplet(image: str, another_image: str) -> tuple:
    query = keras_image.load_img(image)
    query = keras_image.img_to_array(query)
    positive = similar_image(query, 1)[0]
    negative = keras_image.load_img(another_image)
    negative = keras_image.img_to_array(negative)
    negative = similar_image(negative, 1)[0]
    return query, positive, negative


def train_data_generator() -> tuple:
    # todo Modify.
    image_root = r'G:\Workspace\DS&Alg-Project1-Release\data\image'
    images = os.listdir(image_root)
    while True:
        ret = []
        for i in range(BATCH_SIZE / 3):
            sample = random.sample(images, 2)
            sample = list(map(lambda x: image_root + '\\' + x, sample))
            triplet = generate_triplet(sample[0], sample[1])
            ret.extend(list(triplet))
        yield ret


def test():
    a = train_data_generator()
    while True:
        time.sleep(4)
        b = next(a)[0]
        plt.imshow(b[0] / 255)
        plt.show()
        plt.imshow(b[1] / 255)
        plt.show()
        plt.imshow(b[2] / 255)
        plt.show()
