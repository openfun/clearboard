"""image proccessing transforming image to black and white"""
import cv2
import numpy as np


def black_n_white(name, dest):
    """process image on path name to black and white and output in dest"""
    gamma = 0.4
    img_original = cv2.imread(name, 0)
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(img_original, look_up_table)
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(res, 11, offset=10, method="gaussian")
    # res = (res > T).astype("uint8") * 255
    # [changing-contrast-brightness-gamma-correction]

    clahe = cv2.createCLAHE().apply(res)

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(clahe, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.imwrite(dest, thresh)
