"""FastAPI main module for the Clearboard application.
core_address : string, defines the part of the url that wont change between several jitsi-box
origins : string[], url to whitelist and on which the fastapi server should listen (basicly the core address)
"""


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

    # using opencv 2, there are others ways
    img_buffer = imencode('.jpg', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')


def get_image(volume):
    image = volume
    return image_to_base64(image)


@app.get("/photo")
async def photo():
    try:
        if os.path.exists(os.path.abspath('./clearboard/11.jpg')):
            img = Image.open('./clearboard/11.jpg')
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
    try:
        while True:
            await websocket.receive_text()
            if os.path.isfile(os.path.abspath('./clearboard/11.jpg')):
                print('file found')
                if (temps != os.path.getctime('./clearboard/11.jpg')):
                    temps = os.path.getctime('./clearboard/11.jpg')
                    print('new photo to send')
                    await manager.send_personal_message("true", websocket)
                else:
                    await manager.send_personal_message("false", websocket)
            else:
                print('no file found')
                time.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


class Item(BaseModel):
    coord: list


@app.post("/coord")
async def post_coord(c: Item):
    print(c)

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
        cnt = np.array(coordonnees)
        cropped = four_point_transform(image, cnt)

        cv2.imwrite("cropped_reoriented.png", cropped)

    try:
        traitement('./clearboard/11.jpg', c)

    except:
        print('error occured')
