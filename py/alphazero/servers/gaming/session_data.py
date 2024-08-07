from .base_params import BaseParams
from .log_forwarder import LogForwarder

from alphazero.logic.custom_types import ClientId, ClientRole
from games.game_spec import GameSpec
from games.index import get_game_spec
from util.logging_util import get_logger
from util.repo_util import Repo
from util.socket_util import Socket

import os
import socket
import time
from typing import Optional


logger = get_logger()


class SessionData:
    """
    Connecting with the loop controller leads to the creation of a session.

    This class holds various data that is associated with that session.
    """
    def __init__(self, params: BaseParams):
        self._params = params
        self._game = None
        self._game_spec = None
        self._socket: Optional[Socket] = None
        self._client_id: Optional[ClientId] = None

    def init_socket(self):
        addr = (self._params.loop_controller_host, self._params.loop_controller_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(addr)
        self._socket = Socket(sock)

    def send_handshake(self, role: ClientRole, aux: Optional[dict] = None):
        data = {
            'type': 'handshake',
            'role': role.value,
            'start_timestamp': time.time_ns(),
            'cuda_device': self._params.cuda_device,
        }
        if aux is not None:
            data['aux'] = aux

        self.socket.send_json(data)

    def recv_handshake(self, role: ClientRole, log_forwarder: LogForwarder):
        data = self.socket.recv_json(timeout=1)
        assert data['type'] == 'handshake-ack', data

        rejection = data.get('rejection', None)
        if rejection is not None:
            raise Exception(f'Handshake rejected: {rejection}')

        client_id = data['client_id']
        self._game = data['game']
        self._client_id = client_id

        log_forwarder.launch()
        logger.info(f'**** Starting {role.value} ****')
        logger.info(f'Received client id assignment: {client_id}')

    @property
    def socket(self) -> Socket:
        if self._socket is None:
            raise ValueError('loop controller socket not initialized')
        return self._socket

    @property
    def client_id(self) -> ClientId:
        if self._client_id is None:
            raise ValueError('client id not set')
        return self._client_id

    @property
    def game(self) -> str:
        if self._game is None:
            raise ValueError('game not set')
        return self._game

    @property
    def game_spec(self) -> GameSpec:
        if self._game_spec is None:
            self._game_spec = get_game_spec(self.game)
        return self._game_spec
