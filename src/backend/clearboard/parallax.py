import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum)]
    rect[2] = pts[np.argmax(sum)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    #(tl, tr, br, bl) = (top_left, top_right, bottom_right, bottom_left)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def crop(original_img, coordinates, destination):

    image = cv2.imread(original_img)
    cropped = image.copy()
    if coordinates is None or len(coordinates) == 0:
        cv2.imwrite(destination, cropped)
    else:
        try:
            cnt = np.array(coordinates)
            cropped = four_point_transform(image, cnt)
        except Exception as e:
            print(e)
            pass
        cv2.imwrite(destination, cropped)
