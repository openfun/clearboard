import cv2
import numpy as np


def enhance_contrast(name, dest):
    reso = cv2.imread(name)
    cv2.imwrite(dest, 1.4*reso-100)
