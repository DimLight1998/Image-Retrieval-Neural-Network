import os
import random
import cv2
import tqdm
import numpy as np
from matplotlib import pyplot as plt


def get_normalized_hs_histogram(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist = cv2.calcHist([hsv], [0, 1], None, [45, 32], [0, 180, 0, 256])
    # hist = cv2.calcHist([hsv], [0, 1], None, [20, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def get_hs_histogram_emd(hist1: np.ndarray, hist2: np.ndarray):
    assert hist1.shape == hist2.shape
    # Convert 2d histogram to needed form.
    signature1 = np.zeros((hist1.size, 3), dtype=np.float32)
    signature2 = np.zeros((hist1.size, 3), dtype=np.float32)
    for i in range(hist1.shape[0]):
        for j in range(hist1.shape[1]):
            signature1[i * hist1.shape[1] + j, :] = np.array([hist1[i, j], i, j])
            signature2[i * hist1.shape[1] + j, :] = np.array([hist2[i, j], i, j])

    distance = cv2.EMD(signature1, signature2, cv2.DIST_L2)[0]
    return distance

if __name__ == '__main__':
    image_dir = r'G:\Workspace\DS&Alg-Project1-Release\data\image'
    query = 'n01613177_69.JPEG'
    hist_query = get_normalized_hs_histogram(os.path.join(image_dir, query))

    min_distance = None
    min_file = None
    for i in tqdm.tqdm(os.listdir(image_dir)[:200], ascii=True, ncols=200):
        if i == query:
            continue
        hist_i = get_normalized_hs_histogram(os.path.join(image_dir, i))
        distance = get_hs_histogram_emd(hist_i, hist_query)
        if min_distance is None or min_distance > distance:
            min_distance = distance
            min_file = i
    print(min_file)
    #
    #
    # query1 = 'n01613177_69.JPEG'
    # query2 = 'n01613177_70.JPEG'
    #
    # hist1 = get_normalized_hs_histogram(os.path.join(image_dir, query1))
    # hist2 = get_normalized_hs_histogram(os.path.join(image_dir, query2))
    #
    #
    # plt.imshow(hist1, interpolation='nearest')
    # plt.show()
    # plt.imshow(hist2, interpolation='nearest')
    # plt.show()
