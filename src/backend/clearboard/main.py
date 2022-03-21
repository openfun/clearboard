"""FastAPI main module for the Clearboard application.
origins : string[],
url to whitelist and on which the fastapi server should listen (basicly the core address)
"""
import base64
import os
import shutil
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, File, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import cv2  # Import the OpenCV library
import numpy as np
from pydantic import BaseModel

from . import black_n_white, color, config, contrast, parallax

app = FastAPI()

DICO_COORD = {}


class ConnectionManager:
    """Class to monitor websocket communication"""

    def __init__(self):
        self.active_connections: List[(WebSocket, str)] = []

    async def connect(self, websocket: WebSocket, room_name: str):
        """accept websocket sent by the front"""
        await websocket.accept()
        self.active_connections.append((websocket, room_name))

    def disconnect(self, websocket: WebSocket, room_name):
        """disconnect the websocket"""
        self.active_connections.remove((websocket, room_name))

    async def broadcast(self, message: str, room_name: str):
        """given a room name send a meesage to all the clients present in this room name"""
        for connection in self.active_connections:
            if room_name == connection[1]:
                await connection[0].send_text(message)


class Coordinates(BaseModel):
    """given a specific room name, class to define the coordinates for cropping"""

    coord: List[List[str]] = []
    room_name: str


class Process(BaseModel):
    """given a specific room name, class to define the image process used"""

    process: str
    room_name: str


manager = ConnectionManager()


@lru_cache()
def get_settings():
    """get settings form env"""
    return config.Settings()


async def send_message_true_broadcast(room_name):
    """notify all the participants of a room of a new picture"""
    await manager.broadcast("true", room_name)


# Remove env loading
# settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=settings.ORIGINS,
    allow_origins=["https://jitsi-box.com", "https://www.jitsi-box.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/picture")
async def post_picture(file: UploadFile = File(...)):
    """receive image not processed from the jitsi box, not from the student interface"""
    if not file:
        return {"message": "error"}
    path = f"./{file.filename[:-4]}"
    path_original_image = f"{path}/{file.filename}"
    print(file.filename[:-4])
    await send_message_true_broadcast(file.filename[:-4])

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path_original_image, "wb") as original_image:
        shutil.copyfileobj(file.file, original_image)
    return {"message": file.filename}


def image_to_base64(img: np.ndarray) -> bytes:
    """Given a numpy 2D array, returns a JPEG image in base64 format"""
    img_buffer = cv2.imencode(".jpg", img)[1]
    return base64.b64encode(img_buffer).decode("utf-8")


@app.get("/process")
async def get_process(room_name: str, process: str):
    """receive the filter type to use on the image"""

    original_img_path = "./" + room_name + "/" + room_name + ".jpg"
    img_cropped_path = "./" + room_name + "/" + room_name + "cropped.jpg"
    processed_img_path = "./" + room_name + "/" + room_name + process + ".jpg"

    if os.path.exists(os.path.abspath(original_img_path)):
        if room_name in DICO_COORD:
            parallax.crop(original_img_path, DICO_COORD[room_name], img_cropped_path)
            img_to_process = img_cropped_path
        else:
            img_to_process = original_img_path
        if process == "Color":
            color.whiteboard_enhance(img_to_process, processed_img_path)
        elif process == "B&W":
            black_n_white.black_n_white(img_to_process, processed_img_path)
        elif process == "Contrast":
            contrast.enhance_contrast(img_to_process, processed_img_path)
        elif process == "original":
            processed_img_path = img_to_process
        else:
            processed_img_path = img_to_process
        img = cv2.imread(processed_img_path)
        volume = np.asarray(img)
        image = image_to_base64(volume)
        return Response(content=image)


@app.get("/original_photo")
async def photo(room_name: Optional[str] = None):
    """request from front to get the image not processed"""
    original_img_path = "./" + room_name + "/" + room_name + ".jpg"
    if os.path.exists(os.path.abspath(original_img_path)):
        img = cv2.imread(original_img_path)
        volume = np.asarray(img)
        image = image_to_base64(volume)
        return Response(content=image)
    print("original image not found")


@app.websocket("/ws/{room_name}/{id}")
async def websocket_endpoint(websocket: WebSocket, room_name: str):
    """creation of the websocket with the client, given the id and the roomName"""
    await manager.connect(websocket, room_name)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_name)


@app.post("/coord")
async def post_coord(coordinates: Coordinates):
    """receive coordinates from the front, to crop the image"""
    room_name = coordinates.room_name
    coords = [[int(float(k[0])), int(float(k[1]))] for k in coordinates.coord]
    DICO_COORD[room_name] = coords
    await send_message_true_broadcast(room_name)
