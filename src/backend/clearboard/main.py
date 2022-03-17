"""FastAPI main module for the Clearboard application.
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

dico_coord = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[ (WebSocket, str)] = []

    async def connect(self, websocket: WebSocket, roomName : str):
        await websocket.accept()
        self.active_connections.append((websocket, roomName))

    def disconnect(self, websocket: WebSocket, roomName):
        self.active_connections.remove((websocket, roomName))

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, roomName : str):
        for connection in self.active_connections:
            if roomName == connection[1]:
                await connection[0].send_text(message)


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

async def send_message_true_broadcast(roomName):
        await manager.broadcast("true", roomName)
    
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/picture")
async def post_picture(file: UploadFile = File(...)):
    if not file:
        return {"message": "error"}
    else:
        path = f"./{file.filename[:-4]}"
        print(file.filename[:-4])
        await send_message_true_broadcast(file.filename[:-4])
        
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
    global dico_coord
    
    original_img_path = './' + roomName + '/' + roomName + '.jpg'
    img_cropped_path = './' + roomName + '/' + roomName + 'cropped.jpg'
    
    if os.path.exists(os.path.abspath(original_img_path)):
        if roomName in dico_coord.keys():
            parallax.crop(original_img_path, dico_coord[roomName],img_cropped_path)
            img_to_process = img_cropped_path
        else:
            img_to_process = original_img_path
        try:
            processed_img_path = './' + roomName + '/' + roomName + process + ".jpg"
            if process == 'Color':
                color.whiteboard_enhance(cv2.imread(img_to_process), processed_img_path)
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

        except Exception as e:
            print(e)
            print('no file to send')


@app.get("/original_photo")
async def photo(roomName: Optional[str] = None):
    try:
        original_img_path = './' + roomName + '/' + roomName + '.jpg'
        if os.path.exists(os.path.abspath(original_img_path)):
            img = cv2.imread(original_img_path)
            volume = np.asarray(img)
            image = get_image(volume)
            print('original image sent')
            return Response(content=image)
        else:
            print('original image not found')
    except Exception as e:
        print(e)


@app.websocket("/ws/{roomName}/{id}")
async def websocket_endpoint(websocket: WebSocket, roomName: str):
    await manager.connect(websocket, roomName)
    try: 
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, roomName)


@app.post("/coord")
async def post_coord(coordinates: Coordinates):
    global dico_coord
    roomName = coordinates.roomName
    coords = [[int(float(k[0])), int(float(k[1]))] for k in coordinates.coord]
    dico_coord[roomName] = coords
    await send_message_true_broadcast(roomName)
