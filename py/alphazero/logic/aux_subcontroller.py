from alphazero.logic.custom_types import  ClientData, ClientId, ClientType
from alphazero.logic.loop_control_data import LoopControlData
from util.logging_util import get_logger
from util.socket_util import recv_json, send_file, send_json, JsonDict
from util import subprocess_util

from collections import defaultdict
import os
import sqlite3
import threading
from typing import Dict, List


logger = get_logger()


class AuxSubcontroller:
    """
    Shared by other subcontrollers for various functionality.
    """
    def __init__(self, data: LoopControlData):
        self.data = data
        self._pause_ack_events: Dict[ClientId, threading.Event] = defaultdict(threading.Event)

    def handle_disconnect(self, client_data: ClientData):
        logger.info(f'Handling disconnect for {client_data}...')
        self.data.remove_client(client_data.client_id)
        self.data.close_db_conns(threading.get_ident())
        client_data.sock.close()
        logger.info(f'Disconnect complete!')

    def send_asset(self, tgt: str, client_data: ClientData):
        all_assets = [self.data.binary_asset] + self.data.extra_assets
        requested_assets = [a for a in all_assets if a.tgt_path == tgt]
        if len(requested_assets) != 1:
            raise ValueError(f'Invalid asset request: {tgt}')

        asset = requested_assets[0]
        src = asset.src_path
        send_file(client_data.sock, src)

    def accept_client(self, conn: sqlite3.Connection) -> ClientData:
        client_socket, addr = self.data.server_socket.accept()
        ip_address, port = addr

        msg = recv_json(client_socket)
        assert msg['type'] == 'handshake', f'Expected handshake from client, got {msg}'
        role = msg['role']
        client_type = ClientType(role)

        start_timestamp = msg['start_timestamp']
        cuda_device = msg.get('cuda_device', '')

        cursor = conn.cursor()
        cursor.execute('INSERT INTO clients (ip_address, port, role, start_timestamp, cuda_device) VALUES (?, ?, ?, ?, ?)',
                       (ip_address, port, role, start_timestamp, cuda_device)
                       )
        client_id = cursor.lastrowid
        conn.commit()

        client_data = ClientData(
            client_type, client_id, client_socket, start_timestamp, cuda_device)

        self.data.add_client(client_data)

        logger.info(f'Accepted client: {client_data}')
        return client_data

    def handle_pause_ack(self, client_data: ClientData):
        self._pause_ack_events[client_data.client_id].set()

    def pause(self, clients: List[ClientData]):
        if not clients:
            return
        logger.info('Issuing pause...')
        data = {'type': 'pause'}

        for client in clients:
            send_json(client.sock, data)

        for client in clients:
            event = self._pause_ack_events.get(client.client_id)
            event.wait()
            event.clear()
        logger.info('Pause acked!')

    def pause_shared_gpu_self_play_clients(self):
        """
        TODO: pause ratings clients too
        """
        self_play_list = self.data.get_client_data_list(ClientType.SELF_PLAY_WORKER)
        shared_list = [c for c in self_play_list if c.is_on_localhost() and
                       c.cuda_device == self.data.params.cuda_device]
        self.pause(shared_list)

    def reload_weights(self, generation: int):
        clients = self.data.get_client_data_list(ClientType.SELF_PLAY_WORKER)
        if not clients:
            return

        model_filename = self.data.organizer.get_model_filename(generation)
        logger.info('Issuing reload...')

        data = {
            'type': 'reload_weights',
            'model_filename': model_filename,
            'generation': generation,
        }

        for client in clients:
            send_json(client.sock, data)

    def copy_binary_to_bins_dir(self):
        src = self.data.binary_asset.src_path
        tgt = os.path.join(self.data.organizer.bins_dir, self.data.binary_asset.sha256)
        rsync_cmd = ['rsync', '-t', src, tgt]
        subprocess_util.run(rsync_cmd)

    def add_asset_metadata_to_reply(self, reply: JsonDict):
        assets = []
        for asset in [self.data.binary_asset] + self.data.extra_assets:
            assets.append((asset.tgt_path, asset.sha256))
        reply['assets'] = assets

    def setup(self):
        self.data.organizer.makedirs()
        self.copy_binary_to_bins_dir()
        self.data.init_server_socket()
