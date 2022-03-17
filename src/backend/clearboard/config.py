import typing
from pydantic import BaseSettings


class Settings(BaseSettings):
    origins: typing.List = ["example.com", ]

    class Config:
        env_prefix = ''
        env_file = '.env'
        env_file_encoding = 'utf-8'
        arbitrary_types_allowed = True
        case_sensitive = False
