"""FastAPI main module for the Clearboard application."""
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
    data = {"url": coreAddress + customAddress}
    return data
