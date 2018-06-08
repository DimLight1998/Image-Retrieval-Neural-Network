import os
import random
import cv2
import tqdm
import numpy as np
from matplotlib import pyplot as plt

split_n = 4
split_m = 4

def get_normalized_hs_histogram(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist = cv2.calcHist([hsv], [0, 1], None, [3, 4], [0, 180, 0, 256])
    # hist = cv2.calcHist([hsv], [0, 1], None, [20, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def get_normalized_hs_histogram_n_m(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    p = img.shape
    hist = []
    for i in range(split_m):
        hist_row = []
        for j in range(split_n):
            img_split = img[int(p[0] * i / split_m) : int(p[0] * (i + 1) / split_m), int(p[1] * j / split_n) : int(p[1] * (j + 1) / split_n)]
            hsv = cv2.cvtColor(img_split, cv2.COLOR_BGR2HSV)
            hist_split = cv2.calcHist([hsv], [0, 1], None, [3, 4], [0, 180, 0, 256])
            hist_row.append(hist_split)
        hist.append(hist_row)
    hist = np.array(hist)
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

def get_hs_histogram_emd_n_m(hist1: np.ndarray, hist2: np.array):
    assert hist1.shape == hist2.shape
    size = hist1.size
    shape = hist1.shape
    signature1 = np.zeros((size, 5), dtype=np.float32)
    signature2 = np.zeros((size, 5), dtype=np.float32)
    size1 = shape[1] * shape[2] * shape[3]
    size2 = shape[2] * shape[3]
    size3 = shape[3]
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for t in range(shape[3]):
                    signature1[i * size1 + j * size2 + k * size3 + t, :] = np.array([hist1[i, j, k, t], i, j, k, t])
                    signature2[i * size1 + j * size2 + k * size3 + t, :] = np.array([hist2[i, j, k, t], i, j, k, t])

    distance = cv2.EMD(signature1, signature2, cv2.DIST_L2)[0]
    return distance


if __name__ == '__main__':
    image_dir = r'/media/dz/Data/University/2018Spring/Data_Structure_and_Algorithms(2)/DS&Alg-Project1-Release/data/image'
    query = 'n01613177_69.JPEG'
    hist_query = get_normalized_hs_histogram(os.path.join(image_dir, query))

    min_distance = None
    min_file = None
    for i in tqdm.tqdm(os.listdir(image_dir), ascii=True, ncols=200):
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
