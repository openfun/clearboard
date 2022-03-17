"""FastAPI main module for the Clearboard application.
core_address : string, defines the part of the url that wont change between several jitsi-box
origins : string[], url to whitelist and on which the fastapi server should listen (basicly the core address)
"""


import cv2  # Import the OpenCV library
from fastapi import FastAPI, Depends, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from . import blackNwhite, color, config, contrast, super_res
from . import parallax
import shutil
import os

import base64
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel
from starlette.websockets import WebSocket
import time

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


class Coordinates(BaseModel):
    coord: List[List[str]] = []
    roomName: str


class Process(BaseModel):
    process: str
    roomName: str


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
    img_buffer = cv2.imencode('.jpg', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')


def get_image(image):
    return image_to_base64(image)


@app.get("/process")
async def photo(roomName: str, process: str):
    try:
        img_cropped_path = './' + roomName + '/' + roomName + 'cropped.jpg'
        original_img_path = './' + roomName + '/' + roomName + '.jpg'
        processed_img_path = './' + roomName + '/' + roomName + process + ".jpg"
        if os.path.exists(os.path.abspath(img_cropped_path)):
            img_to_process = img_cropped_path
        elif os.path.exists(os.path.abspath(original_img_path)):
            img_to_process = original_img_path

        if process == 'Color':
            color.whiteboard_enhance(cv2.imread(
                img_to_process), processed_img_path)
        elif process == 'B&W':
            blackNwhite.black_n_white(img_to_process, processed_img_path)
        elif process == 'Contrast':
            contrast.enhance_contrast(img_to_process, processed_img_path)
        elif process == 'original':
            processed_img_path = img_to_process
        elif process == 'SuperRes':
            super_res.super_res(img_to_process, processed_img_path, "./clearboard/EDSR/EDSR_x4.pb")
        else:
            processed_img_path = img_to_process
        img = cv2.imread(processed_img_path)
        volume = np.asarray(img)
        image = get_image(volume)
        print('image sent')
        return Response(content=image)

    except:
        print('no file to send')


@app.get("/original_photo")
async def photo(roomName: Optional[str] = None):
    try:
        original_img_path = './' + roomName + '/' + roomName + '.jpg'
        if os.path.exists(os.path.abspath(original_img_path)):
            img = cv2.imread(original_img_path)
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
            print('test')
            roomName = await websocket.receive_text()
            original_img_path = './' + roomName + '/' + roomName + '.jpg'
            img_cropped_path = './' + roomName + '/' + roomName + 'cropped.jpg'

            if (os.path.isfile(os.path.abspath(original_img_path)) and (temps != os.path.getctime(original_img_path))):
                temps = os.path.getctime(original_img_path)
                temps_coord = os.path.getctime('./coord.npy')
                print('change file')
                try:
                    coordinates = np.load("./coord.npy")
                except:
                    coordinates = None
                parallax.crop(original_img_path, coordinates, img_cropped_path)
                print('new photo to send')

                await manager.send_personal_message("true", websocket)

                if (os.path.isfile(os.path.abspath('./coord.npy'))) and (temps_coord != os.path.getctime('./coord.npy')):
                    temps_coord = os.path.getctime('./coord.npy')
                    print('change of coordinates')
                    try:
                        coordinates = np.load("./coord.npy")
                    except:
                        coordinates = None
                    parallax.crop(original_img_path, coordinates, img_cropped_path)
                    print('new photo to send')

                    await manager.send_personal_message("true", websocket)

            else:
                await manager.send_personal_message("false", websocket)

    except WebSocketDisconnect:
        np.save('./coord.npy', None)
        manager.disconnect(websocket)


@app.post("/coord")
async def post_coord(coordinates: Coordinates):
    roomName = coordinates.roomName
    img_cropped_path = './' + roomName + '/' + roomName + 'cropped.jpg'
    original_img_path = './' + roomName + '/' + roomName + '.jpg'
    coords = [[int(float(k[0])), int(float(k[1]))] for k in coordinates.coord]
    np.save('./coord.npy', coords)

    try:
        parallax.crop(original_img_path, coords, img_cropped_path)

    except Exception as e:
        print(e)
