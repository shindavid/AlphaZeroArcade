from .loop_controller_interface import LoopControllerInterface

from alphazero.logic.custom_types import ClientConnection, ClientRole, GpuInfo
from util.logging_util import get_logger
from util.socket_util import recv_json, Socket

import threading
from typing import List, Optional


logger = get_logger()


class ClientConnectionManager:
    def __init__(self, controller: LoopControllerInterface):
        self._controller = controller
        self._connections: List[ClientConnection] = []
        self._lock = threading.Lock()

    def get(self, role: ClientRole, gpu_info: Optional[GpuInfo]=None) -> List[ClientConnection]:
        """
        Returns a list of all connections that match the given type/gpu_info.
        """
        with self._lock:
            conns = list(self._connections)

        conns = [c for c in conns if c.client_role == role]
        if gpu_info is not None:
            conns = [c for c in conns if c.client_gpu_info == gpu_info]
        return conns

    def remove(self, conn: ClientConnection):
        with self._lock:
            self._connections = [c for c in self._connections if c.client_id != conn.client_id]

    def start(self):
        logger.info(f'Listening for connections on port {self._controller.params.port}...')
        threading.Thread(target=self._accept_connections, name='accept_connections',
                         daemon=True).start()

    def _accept_connections(self):
        try:
            while True:
                conn = self._add_connection()
                self._controller.handle_new_client_connnection(conn)
        except:
            logger.error('Exception in accept_connections():', exc_info=True)
            self._controller.request_shutdown(1)

    def _add_connection(self) -> ClientConnection:
        db_conn = self._controller.clients_db_conn_pool.get_connection()
        client_socket, addr = self._controller.socket.accept()
        ip_address, port = addr

        msg = recv_json(client_socket)
        logger.debug(f'Received json message: {msg}')
        assert msg['type'] == 'handshake', f'Expected handshake from client, got {msg}'
        role = msg['role']
        client_role = ClientRole(role)

        start_timestamp = msg['start_timestamp']
        cuda_device = msg.get('cuda_device', '')

        gpu_info = GpuInfo(ip_address, cuda_device)
        clashing_conns = self.get(client_role, gpu_info)
        if clashing_conns:
            logger.warn(f'Rejection connection due to role/gpu clash: {clashing_conns[0]}')

            reply = {
                'type': 'handshake-ack',
                'rejection': 'connection of same role/cuda-device from same ip already exists',
            }
            tmp_socket = Socket(client_socket)
            tmp_socket.send_json(reply)
            tmp_socket.close()
            return

        with self._lock:
            cursor = db_conn.cursor()
            cursor.execute('INSERT INTO clients (ip_address, port, role, start_timestamp, cuda_device) VALUES (?, ?, ?, ?, ?)',
                            (ip_address, port, role, start_timestamp, cuda_device)
                            )
            client_id = cursor.lastrowid
            db_conn.commit()

        conn = ClientConnection(client_role, client_id, Socket(client_socket), start_timestamp,
                                cuda_device)

        with self._lock:
            self._connections.append(conn)

        logger.info(f'Added connection: {conn}')
        return conn
