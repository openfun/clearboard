"""FastAPI main module for the Clearboard application.
some env vars are needed to configure the server, see README for further informations
"""
import base64
import os
import shutil
from functools import lru_cache
from typing import List

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

import cv2  # Import the OpenCV library
import numpy as np

from . import config, coord_loader
from .filters import black_n_white, color, contrast, parallax
from .models import Coordinates


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


# Getting env vars


@lru_cache()
def get_settings():
    """get settings form env"""
    return config.Settings()


settings = get_settings()
# Websocket
manager = ConnectionManager()
# FastAPI server
app = FastAPI()
# add Middleware settings to open connection with front
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS.split(";"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(img: np.ndarray) -> bytes:
    """Given a numpy 2D array, returns a JPEG image in base64 format"""
    img_buffer = cv2.imencode(".jpg", img)[1]
    return base64.b64encode(img_buffer).decode("utf-8")


async def send_message_true_broadcast(room_name):
    """notify all the participants of a room of a new picture"""
    await manager.broadcast("true", room_name)


@app.post("/picture")
async def post_picture(file: UploadFile = File(...)):
    """receive image not filtered from the jitsi box, not from the student interface"""
    if not file:
        raise HTTPException(status_code=502, detail="Picture not received")
    path = f"{settings.MEDIA_ROOT}/{file.filename[:-4]}"
    path_original_image = f"{path}/{file.filename}"
    await send_message_true_broadcast(file.filename[:-4])

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path_original_image, "wb") as original_image:
        shutil.copyfileobj(file.file, original_image)
    return {"message": file.filename}


@app.get("/filter")
async def get_process(room_name: str, filter: str):
    """receive the filter type to use on the image"""

    original_img_path = settings.MEDIA_ROOT + "/" + room_name + "/" + room_name + ".jpg"
    img_cropped_path = (
        settings.MEDIA_ROOT + "/" + room_name + "/" + room_name + "cropped.jpg"
    )
    coord_path = settings.MEDIA_ROOT + "/" + room_name + "/coord.txt"
    filtered_img_path = (
        settings.MEDIA_ROOT + "/" + room_name + "/" + room_name + filter + ".jpg"
    )

    if os.path.exists(os.path.abspath(original_img_path)):
        if os.path.exists(os.path.abspath(coord_path)):
            parallax.crop(
                original_img_path, coord_loader.get_coords(coord_path), img_cropped_path
            )
            img_to_filter = img_cropped_path
        else:
            img_to_filter = original_img_path
        if filter == "Color":
            color.whiteboard_enhance(img_to_filter, filtered_img_path)
        elif filter == "B&W":
            black_n_white.black_n_white(img_to_filter, filtered_img_path)
        elif filter == "Contrast":
            contrast.enhance_contrast(img_to_filter, filtered_img_path)
        elif filter == "original":
            filtered_img_path = img_to_filter
        else:
            filtered_img_path = img_to_filter
        return Response(
            content=image_to_base64(np.asarray(cv2.imread(filtered_img_path)))
        )
    else:
        raise HTTPException(status_code=404, detail="No image to filter")


@app.get("/original_photo")
async def photo(room_name: str = None):
    """request from front to get the image not processed"""
    if room_name == None:
        raise HTTPException(status_code=404, detail="No room name given")
    else:
        original_img_path = (
            settings.MEDIA_ROOT + "/" + room_name + "/" + room_name + ".jpg"
        )
        if os.path.exists(os.path.abspath(original_img_path)):
            return Response(
                content=image_to_base64(np.asarray(cv2.imread(original_img_path)))
            )
        else:
            raise HTTPException(status_code=404, detail="No room name given")


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
    coords = [[int(float(k[0])), int(float(k[1]))] for k in coordinates.coord]
    coord_dir_path = settings.MEDIA_ROOT + "/" + coordinates.room_name
    if not os.path.exists(coord_dir_path):
        os.makedirs(coord_dir_path)
    coord_loader.save_coords(coord_dir_path + "/coord.txt", coords)
    await send_message_true_broadcast(coordinates.room_name)
