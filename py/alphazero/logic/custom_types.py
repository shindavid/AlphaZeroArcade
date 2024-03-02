from enum import Enum


Generation = int


class ChildThreadError(Exception):
    pass


class ClientType(Enum):
    SELF_PLAY_SERVER = 'self-play-server'
    SELF_PLAY = 'self-play'
