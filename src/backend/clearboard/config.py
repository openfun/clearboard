"""Env configuration module"""
import typing

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Define fastapi settings"""

    ORIGINS: typing.List = []
    MEDIA_ROOT: str

    class Config:
        """Define fastapi env configuration"""

        env_prefix = ""
        env_file = "../../env.d/development.dist"
        env_file_encoding = "utf-8"
        arbitrary_types_allowed = True
        case_sensitive = False
