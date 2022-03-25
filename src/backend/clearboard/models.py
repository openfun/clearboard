"""Models defined for clearboard server"""
from typing import List
from pydantic import BaseModel


class Coordinates(BaseModel):
    """given a specific room name, class to define the coordinates for cropping"""

    coord: List[List[str]] = []
    room_name: str
