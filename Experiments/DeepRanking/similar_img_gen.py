from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np


def sometimes(aug): return iaa.Sometimes(0.5, aug)


seq = iaa.Sequential([
    iaa.Fliplr(0.15),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8))
], random_order=True)


def similar_image(image: np.ndarray, n: int) -> np.ndarray:
    ret = np.zeros((n,) + image.shape)
    for i in range(n):
        ret[i] = seq.augment_image(image)
    return ret
