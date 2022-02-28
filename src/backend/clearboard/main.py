"""FastAPI main module for the Clearboard application.
coreAddress : string, defines the part of the url that wont change between several jitsi-box
origins : string[], url to whitelist and on which the fastapi server should listen (basicly the core address)
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
coreAddress = "http://localhost:3000/"

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/getPolicy")
def getPolicy(customAddress: str = "dty"):
    """customAddress: string, part of the url that identify one meeting from another """
    data = {"url": coreAddress + customAddress}
    return data
