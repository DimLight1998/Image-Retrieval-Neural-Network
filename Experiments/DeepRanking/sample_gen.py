import json
import os
import random
import time

import numpy as np
from keras.preprocessing import image as keras_image
from Experiments.DeepRanking.similar_img_gen import similar_image as similar_image
from Experiments.DeepRanking.deep_rank import BATCH_SIZE


def generate_triplet(image: str, another_image: str) -> tuple:
    query = keras_image.load_img(image, target_size=(225, 225))
    query = keras_image.img_to_array(query)
    positive = similar_image(query, 1)[0]
    negative = keras_image.load_img(another_image, target_size=(225, 225))
    negative = keras_image.img_to_array(negative)
    negative = similar_image(negative, 1)[0]
    return query, positive, negative


def train_data_generator() -> tuple:
    # todo Modify.
    image_root = r'G:\Workspace\DS&Alg-Project1-Release\data\image'
    images = os.listdir(image_root)
    while True:
        ret_x = np.ndarray((BATCH_SIZE, 225, 225, 3))
        for i in range(int(BATCH_SIZE / 3)):
            sample = random.sample(images, 2)
            sample = list(map(lambda x: image_root + '\\' + x, sample))
            triplet = generate_triplet(sample[0], sample[1])
            ret_x[i + 0] = triplet[0]
            ret_x[i + 1] = triplet[1]
            ret_x[i + 2] = triplet[2]
        ret_y = np.zeros((BATCH_SIZE))
        yield ret_x, ret_y
