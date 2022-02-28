"""image proccessing cropping and reorienting the image"""
import cv2
import numpy as np


def order_points(pts):
    """render a list of the coordinates of
    the 4 points selected according to this format (tl, tr, br, bl) = rect"""
    rect = np.zeros((4, 2), dtype="float32")
    sum_pts = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum_pts)]
    rect[2] = pts[np.argmax(sum_pts)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """return the image cropped"""
    rect = order_points(pts)
    width_first = np.sqrt(
        ((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2)
    )
    width_second = np.sqrt(
        ((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2)
    )
    max_width = max(int(width_first), int(width_second))
    height_first = np.sqrt(
        ((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2)
    )
    height_second = np.sqrt(
        ((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2)
    )
    max_height = max(int(height_first), int(height_second))
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    view_transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, view_transform, (max_width, max_height))
    return warped


def crop(original_img, coordinates, destination):
    """main function called to crop an image in function of the coordinates
    of the points given, then it saves this file with the path given by destination"""
    image = cv2.imread(original_img)
    cropped = image.copy()
    if coordinates is None or len(coordinates) == 0:
        cv2.imwrite(destination, cropped)
    else:
        cnt = np.array(coordinates)
        cropped = four_point_transform(image, cnt)
        cv2.imwrite(destination, cropped)
