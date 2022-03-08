"""FastAPI main module for the Clearboard application.
core_address : string, defines the part of the url that wont change between several jitsi-box
origins : string[], url to whitelist and on which the fastapi server should listen (basicly the core address)
"""
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


        
@app.websocket("/ws11")
async def websocket_endpoint(websocket: WebSocket):
    print("started")
    #await websocket.accept()
    await manager.connect(websocket)
    temps = 0
    cnt = 0
    try:
        while True:
            time.sleep(1)
            cnt +=1
            if (temps != os.path.getctime('./clearboard/11.jpg')):
                img = Image.open('./clearboard/11.jpg')
                time.sleep(1)
                temps = os.path.getctime('./clearboard/11.jpg')
                volume = np.asarray(img)
                image = get_image(volume)
                await websocket.send_bytes(image)
                print('image sent')
            else:
                print("pas de changement")
                if cnt == 10:
                    cnt = 0
                    print('cnt')
    except WebSocketDisconnect:
            print('deco')
            manager.disconnect(websocket)
             
def image_to_base64(img: np.ndarray) -> bytes:
    """ Given a numpy 2D array, returns a JPEG image in base64 format """

    # using opencv 2, there are others ways
    img_buffer = imencode('.jpg', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')
    
def get_image(volume):
    image = volume
    return image_to_base64(image)


### test
async def _alive_task(websocket: WebSocket):
    try:
        await websocket.receive_text()
        asyncio.sleep(4)
    except (WebSocketDisconnect, ConnectionClosedError):
        pass
        
async def _send_data(websocket: WebSocket):
    try:
        while True:
            img = Image.open('./clearboard/11.jpg')
            volume = np.asarray(img)
            image = get_image(volume)
            await websocket.send_bytes(image)
            asyncio.sleep(4)
            print('image sent')
    except (WebSocketDisconnect, ConnectionClosedError):
        print("error")
        pass


@app.websocket("/ws")
async def handle_something(websocket: WebSocket):
    await websocket.accept()
    
    loop = asyncio.get_running_loop()
    alive_task = loop.create_task(
        _alive_task(websocket),
        name=f"WS alive check: {websocket.client}",
    )
    send_task: asyncio.Task = loop.create_task(
        _send_data(websocket),
        name=f"WS data sending: {websocket.client}",
    )
    
    alive_task.add_done_callback(send_task.cancel)
    send_task.add_done_callback(alive_task.cancel)
    
    await asyncio.wait({alive_task, send_task})


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