"""image proccessing transforming image to black and white"""
import cv2
import numpy as np


def black_n_white(name, dest):
    """process image on path name to black and white and output in dest"""
    gamma = 0.8

    img_original = cv2.imread(name, 0)
    values = np.arange(0, 256)
    look_up_table = np.uint8(255 * np.power((values / 255.0), gamma))
    res = cv2.LUT(img_original, look_up_table)

    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16, 16)).apply(res)

    thresh = cv2.adaptiveThreshold(
        clahe, 190, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 12
    )
    cv2.imwrite(dest, thresh)
