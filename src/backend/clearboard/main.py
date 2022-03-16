"""FastAPI main module for the Clearboard application.
core_address : string, defines the part of the url that wont change between several jitsi-box
origins : string[], url to whitelist and on which the fastapi server should listen (basicly the core address)
"""


from importlib.resources import path
from locale import strcoll
import cv2  # Import the OpenCV library
from fastapi import FastAPI, Depends, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from . import config
from . import parallax, process1, process2, process3, process4
import shutil
import os

import base64
from typing import List, Optional
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


class Process(BaseModel):
    process: str
    roomName: str


@app.get("/process")
async def photo(roomName: str, process: str):
    try:
        path_img_cropped = './' + roomName + '/' + roomName + 'cropped.jpg'
        original_photo_path = './' + roomName + '/' + roomName + '.jpg'
        after = './' + roomName + '/' + roomName + process+".jpg"
        if os.path.exists(os.path.abspath(path_img_cropped)):
            before = path_img_cropped
        elif os.path.exists(os.path.abspath(original_photo_path)):
            before = original_photo_path

        if process == 'Color':
            process1.whiteboard_enhance(cv2.imread(
                before), after)
        elif process == 'B&W':
            process2.black_n_white(before, after)
        elif process == 'Contrast':
            process3.enhance_contrast(before, after)
        elif process == 'original':
            after = before
        elif process == 'SuperRes':
            process4.super_res(before, after, "./clearboard/EDSR/EDSR_x4.pb")
        else:
            after = before
        img = cv2.imread(after)
        volume = np.asarray(img)
        image = get_image(volume)
        print('image sent')
        return Response(content=image)

    except:
        print('no file to send')


@app.get("/original_photo")
async def photo(roomName: Optional[str] = None):
    try:
        original_photo_path = './' + roomName + '/' + roomName + '.jpg'
        if os.path.exists(os.path.abspath(original_photo_path)):
            img = cv2.imread(original_photo_path)
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
            original_photo_path = './' + roomName + '/' + roomName + '.jpg'
            path_img_cropped = './' + roomName + '/' + roomName + 'cropped.jpg'

            if (os.path.isfile(os.path.abspath(original_photo_path)) and (temps != os.path.getctime(original_photo_path))):
                temps = os.path.getctime(original_photo_path)
                temps_coord = os.path.getctime('./coord.npy')
                print('change file')
                try:
                    coordinates = np.load("./coord.npy")
                except:
                    coordinates = None
                parallax.crop(original_photo_path, coordinates, path_img_cropped)
                print('new photo to send')

                await manager.send_personal_message("true", websocket)

                if (os.path.isfile(os.path.abspath('./coord.npy'))) and (temps_coord != os.path.getctime('./coord.npy')):
                    temps_coord = os.path.getctime('./coord.npy')
                    print('change of coordinates')
                    try:
                        coordinates = np.load("./coord.npy")
                    except:
                        coordinates = None
                    parallax.crop(original_photo_path, coordinates, path_img_cropped)
                    print('new photo to send')

                    await manager.send_personal_message("true", websocket)

            else:
                await manager.send_personal_message("false", websocket)

    except WebSocketDisconnect:
        np.save('./coord.npy', None)
        manager.disconnect(websocket)


class Coordinates(BaseModel):
    coord: List[List[str]] = []
    roomName: str


@app.post("/coord")
async def post_coord(coordinates: Coordinates):
    roomName = coordinates.roomName
    path_img_cropped = './' + roomName + '/' + roomName + 'cropped.jpg'
    original_photo_path = './' + roomName + '/' + roomName + '.jpg'
    c = coordinates.coord
    co = [[int(float(k[0])), int(float(k[1]))] for k in c]
    start = time.time()
    np.save('./coord.npy', co)
    end = time.time()
    print('temps np save', end - start)

    try:
        start = time.time()
        parallax.crop(original_photo_path, co, path_img_cropped)
        end = time.time()
        print('temps traitement', end - start)

    except Exception as e:
        print(e)
