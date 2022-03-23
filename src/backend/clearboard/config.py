"""Env configuration module"""
import typing

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Define fastapi settings"""

    ORIGINS: str
    MEDIA_ROOT: str

    class Config:
        """Define fastapi env configuration"""

        arbitrary_types_allowed = True
        case_sensitive = False
