"""image proccessing cropping and reorienting the image"""
import cv2
import numpy as np


def order_points(points):
    """
    input : points : list of 4 points that defines 4 corners
    return a list of the coordinates of
    the 4 points selected formated to this format
    (top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner)"""

    ordered_points = np.zeros((4, 2), dtype="float32")
    # sorting points by first coordinate
    abs_sorted = sorted(points, key=lambda point: point[0])

    # ordering 2 left points
    if abs_sorted[0][1] > abs_sorted[1][1]:
        ordered_points[3] = abs_sorted[0]
        ordered_points[0] = abs_sorted[1]
    else:
        ordered_points[3] = abs_sorted[1]
        ordered_points[0] = abs_sorted[0]

    # ordering 2 right points
    if abs_sorted[2][1] > abs_sorted[3][1]:
        ordered_points[1] = abs_sorted[3]
        ordered_points[2] = abs_sorted[2]
    else:
        ordered_points[1] = abs_sorted[2]
        ordered_points[2] = abs_sorted[3]

    return ordered_points


def four_point_transform(image, points):
    """return the image cropped"""
    ordered_points = order_points(points)
    width_first = np.sqrt(
        ((ordered_points[2][0] - ordered_points[3][0]) ** 2)
        + ((ordered_points[2][1] - ordered_points[3][1]) ** 2)
    )
    width_second = np.sqrt(
        ((ordered_points[1][0] - ordered_points[0][0]) ** 2)
        + ((ordered_points[1][1] - ordered_points[0][1]) ** 2)
    )
    # width of the image once the paralaxe is applied
    max_width = max(int(width_first), int(width_second))
    height_first = np.sqrt(
        ((ordered_points[1][0] - ordered_points[2][0]) ** 2)
        + ((ordered_points[1][1] - ordered_points[2][1]) ** 2)
    )
    height_second = np.sqrt(
        ((ordered_points[0][0] - ordered_points[3][0]) ** 2)
        + ((ordered_points[0][1] - ordered_points[3][1]) ** 2)
    )
    # height of the image once the paralaxe is applied
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
    view_transform = cv2.getPerspectiveTransform(ordered_points, dst)
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
        points = np.array(coordinates)
        cropped = four_point_transform(image, points)
        cv2.imwrite(destination, cropped)
