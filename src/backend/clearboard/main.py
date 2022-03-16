"""FastAPI main module for the Clearboard application.
core_address : string, defines the part of the url that wont change between several jitsi-box
origins : string[], url to whitelist and on which the fastapi server should listen (basicly the core address)
"""
from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from . import config
import shutil
import os

app = FastAPI()


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
