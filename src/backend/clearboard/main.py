"""FastAPI main module for the Clearboard application.
core_address : string, defines the part of the url that wont change between several jitsi-box
origins : string[], url to whitelist and on which the fastapi server should listen (basicly the core address)
"""


import string
import cv2  # Import the OpenCV library
from fastapi import FastAPI, Depends, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from . import config
import shutil
import os

import base64
from typing import List
from PIL import Image
from cv2 import imencode
from pydantic import BaseModel
from starlette.websockets import WebSocket, WebSocketState
from starlette.responses import StreamingResponse
import time
import asyncio

import numpy as np
from . import config

app = FastAPI()
np.save('./coord.npy', None)
img_cropped = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@lru_cache()
def get_settings():
    return config.Settings()


settings = get_settings()
print(settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/policy")
async def get_policy(custom_address: str = "picture", settings: config.Settings = Depends(get_settings)):
    """custom_address: string, part of the url that identify one meeting from another """
    data = {"url": f"{settings.core_address}{custom_address}"}
    return data


@app.post("/picture")
async def post_picture(file: UploadFile = File(...)):
    if not file:
        return {"message": "error"}
    else:
        path = f"./{file.filename[:-4]}"
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/{file.filename}", 'wb') as f:
            shutil.copyfileobj(file.file, f)
        return {"message": file.filename}


def image_to_base64(img: np.ndarray) -> bytes:
    """ Given a numpy 2D array, returns a JPEG image in base64 format """
    img_buffer = imencode('.jpg', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')


def get_image(volume):
    image = volume
    return image_to_base64(image)


@app.get("/photo")
async def photo(process=""):
    try:
        if os.path.exists(os.path.abspath('./cropped_reoriented.jpg')):
            img = Image.open('./cropped_reoriented.jpg')
            volume = np.asarray(img)
            image = get_image(volume)
            print('image sent')
            if process == "process1":
                print("process 1 selected")
                whiteboard_enhance(cv2.imread("./cropped_reoriented.jpg"))
                img = Image.open('./cropped_reoriented.jpg')
                volume = np.asarray(img)
                image = get_image(volume)
            elif process == "process2":
                print("process 2 selected")
                return 'process 2 selected'
            elif process == "process3":
                print("process 3 selected")
                return 'process 3 selected'
            elif process == "process4":
                print("process 4 selected")
                return 'process 4 selected'
            return Response(content=image)
        else:
            print('file not found')
            time.sleep(1)
    except:
        print('no file to send')


@app.get("/original_photo")
async def photo():
    try:
        if os.path.exists(os.path.abspath('./clearboard/img_test.jpg')):
            img = Image.open('./clearboard/img_test.jpg')
            volume = np.asarray(img)
            image = get_image(volume)
            print('image sent')
            return Response(content=image)
        else:
            print('file not found')
            time.sleep(1)
    except:
        print('no file to send')


@app.websocket("/ws1")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    temps = 0
    temps_coord = 0
    try:
        while True:
            await websocket.receive_text()
            if (os.path.isfile(os.path.abspath('./clearboard/img_test.jpg')) and (temps != os.path.getctime('./clearboard/img_test.jpg'))):
                temps = os.path.getctime('./clearboard/img_test.jpg')
                temps_coord = os.path.getctime('./coord.npy')
                print('change file')
                try:
                    c = np.load("./coord.npy")
                except:
                    c = None
                traitement("./clearboard/img_test.jpg", c)

                print('new photo to send')

                await manager.send_personal_message("true", websocket)

            elif (os.path.isfile(os.path.abspath('./coord.npy'))) and (temps_coord != os.path.getctime('./coord.npy')):
                temps_coord = os.path.getctime('./coord.npy')
                print('change of coordinates')
                try:
                    c = np.load("./coord.npy")
                except:
                    c = None
                traitement("./clearboard/img_test.jpg", c)
                print('new photo to send')

                await manager.send_personal_message("true", websocket)

            else:
                await manager.send_personal_message("false", websocket)

    except WebSocketDisconnect:
        np.save('./coord.npy', None)
        manager.disconnect(websocket)


class Coordinates(BaseModel):
    coord: List[List[str]] = []


def traitement(imageNT, coordonnees):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(image, pts):
        rect = order_points(pts)
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

    image = cv2.imread(imageNT)
    cropped = image.copy()
    try:
        cnt = np.array(coordonnees)
        cropped = four_point_transform(image, cnt)
    except:
        pass

    start = time.time()
    cv2.imwrite("./cropped_reoriented.jpg", cropped)
    end = time.time()
    print('temps imwrite save', end - start)


def normalize_kernel(kernel, k_width, k_height, scaling_factor=1.0):
    '''Zero-summing normalize kernel'''

    K_EPS = 1.0e-12
    # positive and negative sum of kernel values
    pos_range, neg_range = 0, 0
    for i in range(k_width * k_height):
        if abs(kernel[i]) < K_EPS:
            kernel[i] = 0.0
        if kernel[i] < 0:
            neg_range += kernel[i]
        else:
            pos_range += kernel[i]

    # scaling factor for positive and negative range
    pos_scale, neg_scale = pos_range, -neg_range
    if abs(pos_range) >= K_EPS:
        pos_scale = pos_range
    else:
        pos_sacle = 1.0
    if abs(neg_range) >= K_EPS:
        neg_scale = 1.0
    else:
        neg_scale = -neg_range

    pos_scale = scaling_factor / pos_scale
    neg_scale = scaling_factor / neg_scale

    # scale kernel values for zero-summing kernel
    for i in range(k_width * k_height):
        if (not np.nan == kernel[i]):
            kernel[i] *= pos_scale if kernel[i] >= 0 else neg_scale

    return kernel


def dog(img, k_size, sigma_1, sigma_2):
    '''Difference of Gaussian by subtracting kernel 1 and kernel 2'''

    k_width = k_height = k_size
    x = y = (k_width - 1) // 2
    kernel = np.zeros(k_width * k_height)

    # first gaussian kernal
    if sigma_1 > 0:
        co_1 = 1 / (2 * sigma_1 * sigma_1)
        co_2 = 1 / (2 * np.pi * sigma_1 * sigma_1)
        i = 0
        for v in range(-y, y + 1):
            for u in range(-x, x + 1):
                kernel[i] = np.exp(-(u*u + v*v) * co_1) * co_2
                i += 1
    # unity kernel
    else:
        kernel[x + y * k_width] = 1.0

    # subtract second gaussian from kernel
    if sigma_2 > 0:
        co_1 = 1 / (2 * sigma_2 * sigma_2)
        co_2 = 1 / (2 * np.pi * sigma_2 * sigma_2)
        i = 0
        for v in range(-y, y + 1):
            for u in range(-x, x + 1):
                kernel[i] -= np.exp(-(u*u + v*v) * co_1) * co_2
                i += 1
    # unity kernel
    else:
        kernel[x + y * k_width] -= 1.0

    # zero-normalize scling kernel with scaling factor 1.0
    norm_kernel = normalize_kernel(
        kernel, k_width, k_height, scaling_factor=1.0)

    # apply filter with norm_kernel
    return cv2.filter2D(img, -1, norm_kernel.reshape(k_width, k_height))


def negate(img):
    '''Negative of image'''

    return cv2.bitwise_not(img)


def get_black_white_indices(hist, tot_count, black_count, white_count):
    '''Blacking and Whiting out indices same as color balance'''

    black_ind = 0
    white_ind = 255
    co = 0
    for i in range(len(hist)):
        co += hist[i]
        if co > black_count:
            black_ind = i
            break

    co = 0
    for i in range(len(hist) - 1, -1, -1):
        co += hist[i]
        if co > (tot_count - white_count):
            white_ind = i
            break

    return [black_ind, white_ind]


def contrast_stretch(img, black_point, white_point):
    '''Contrast stretch image with black and white cap'''

    tot_count = img.shape[0] * img.shape[1]
    black_count = tot_count * black_point / 100
    white_count = tot_count * white_point / 100
    ch_hists = []
    # calculate histogram for each channel
    for ch in cv2.split(img):
        ch_hists.append(cv2.calcHist([ch], [0], None, [
                        256], (0, 256)).flatten().tolist())

    # get black and white percentage indices
    black_white_indices = []
    for hist in ch_hists:
        black_white_indices.append(get_black_white_indices(
            hist, tot_count, black_count, white_count))

    stretch_map = np.zeros((3, 256), dtype='uint8')

    # stretch histogram
    for curr_ch in range(len(black_white_indices)):
        black_ind, white_ind = black_white_indices[curr_ch]
        for i in range(stretch_map.shape[1]):
            if i < black_ind:
                stretch_map[curr_ch][i] = 0
            else:
                if i > white_ind:
                    stretch_map[curr_ch][i] = 255
                else:
                    if (white_ind - black_ind) > 0:
                        stretch_map[curr_ch][i] = round(
                            (i - black_ind) / (white_ind - black_ind)) * 255
                    else:
                        stretch_map[curr_ch][i] = 0

    # stretch image
    ch_stretch = []
    for i, ch in enumerate(cv2.split(img)):
        ch_stretch.append(cv2.LUT(ch, stretch_map[i]))

    return cv2.merge(ch_stretch)


def fast_gaussian_blur(img, ksize, sigma):
    '''Gussian blur using linear separable property of Gaussian distribution'''

    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    return cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)


def gamma(img, gamma_value):
    '''Gamma correction of image'''

    i_gamma = 1 / gamma_value
    lut = np.array([((i / 255) ** i_gamma) *
                   255 for i in np.arange(0, 256)], dtype='uint8')
    return cv2.LUT(img, lut)


def color_balance(img, low_per, high_per):
    '''Contrast stretch image by histogram equilization with black and white cap'''

    tot_pix = img.shape[1] * img.shape[0]
    # no.of pixels to black-out and white-out
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    cs_img = []
    # for each channel, apply contrast-stretch
    for ch in cv2.split(img):
        # cummulative histogram sum of channel
        cum_hist_sum = np.cumsum(cv2.calcHist(
            [ch], [0], None, [256], (0, 256)))

        # find indices for blacking and whiting out pixels
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if (li == hi):
            cs_img.append(ch)
            continue
        # lut with min-max normalization for [0-255] bins
        lut = np.array([0 if i < li
                        else (255 if i > hi else round((i - li) / (hi - li) * 255))
                        for i in np.arange(0, 256)], dtype='uint8')
        # constrast-stretch channel
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)

    return cv2.merge(cs_img)


def whiteboard_enhance(img):
    '''Enhance Whiteboard image'''

    dog_k_size, dog_sigma_1, dog_sigma_2 = 15, 100, 0
    cs_black_per, cs_white_per = 2, 99.5
    gauss_k_size, gauss_sigma = 3, 1
    gamma_value = 1.1
    cb_black_per, cb_white_per = 2, 1

    # Difference of Gaussian (DoG)
    dog_img = dog(img, dog_k_size, dog_sigma_1, dog_sigma_2)
    # Negative of image
    negative_img = negate(dog_img)
    # Contrast Stretch (CS)
    contrast_stretch_img = contrast_stretch(
        negative_img, cs_black_per, cs_white_per)
    # Gaussian Blur
    blur_img = fast_gaussian_blur(
        contrast_stretch_img, gauss_k_size, gauss_sigma)
    # Gamma Correction
    gamma_img = gamma(blur_img, gamma_value)
    # Color Balance (CB) (also Contrast Stretch)
    color_balanced_img = color_balance(gamma_img, cb_black_per, cb_white_per)

    cv2.imwrite("./cropped_reoriented.jpg", color_balanced_img)


@app.post("/coord")
async def post_coord(coordinates: Coordinates):
    c = coordinates.coord
    co = [[int(float(k[0])), int(float(k[1]))] for k in c]
    start = time.time()
    np.save('./coord.npy', co)
    end = time.time()
    print('temps np save', end - start)

    try:
        start = time.time()
        traitement('./clearboard/img_test.jpg', co)
        end = time.time()
        print('temps traitement', end - start)

    except Exception as e:
        print(e)
