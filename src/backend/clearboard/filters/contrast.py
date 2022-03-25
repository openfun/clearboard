"""image proccessing enhancing image contrast"""
import cv2


def enhance_contrast(name, dest):
    """process image on path name to enhance contrast and output in dest"""
    img = cv2.imread(name)
    cv2.imwrite(dest, 1.4 * img - 100)
