"""FastAPI main module for the Clearboard application."""
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    """Hello world API endpoint."""
    return {"Hello": "World"}
