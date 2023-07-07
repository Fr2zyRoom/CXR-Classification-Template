import numpy as np
import cv2


def _cosine_gamma_func(x, a):
    return 1 + a*np.cos(x/255*np.pi)


def cosine_gamma_correction(img):
    table = [((i / 255) ** (1/_cosine_gamma_func(i, .5))) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(img, table)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def clahe(img):
    return _clahe.apply(img)


def gamma_correction(img, gamma=.25):
    inv_gamma = 1 / gamma
    
    return np.uint8(((img / 255) ** inv_gamma) * 255)


def gray2rgb(img):
    return np.stack([img]*3,-1)