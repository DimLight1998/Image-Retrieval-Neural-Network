from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
from matplotlib import pyplot as plt


def sometimes(aug): return iaa.Sometimes(0.5, aug)


seq = iaa.Sequential([
    iaa.Fliplr(0.02),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.3, iaa.Superpixels(0.5, 125)),
    iaa.Add((-5, 5), True),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-10, 10),
        shear=(-5, 5)),
    iaa.OneOf([
        iaa.PerspectiveTransform(0.05),
        iaa.PiecewiseAffine(0.02)])
], random_order=True)


def similar_image(image: np.ndarray, n: int) -> np.ndarray:
    ret = np.zeros((n,) + image.shape)
    for i in range(n):
        ret[i] = seq.augment_image(image)
    return ret
