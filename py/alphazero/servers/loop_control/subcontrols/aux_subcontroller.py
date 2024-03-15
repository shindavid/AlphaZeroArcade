from alphazero.logic.custom_types import ClientData, ClientId, ClientType, Generation, \
    NewModelSubscriber
from alphazero.servers.loop_control.loop_control_data import LoopControlData
from util.logging_util import get_logger
from util.socket_util import recv_json, send_file, send_json, JsonDict, Socket, \
    SocketRecvException, SocketSendException
from util import subprocess_util

import os
import threading
from typing import Callable, List, Optional, Set


logger = get_logger()


MsgHandler = Callable[[ClientData, JsonDict], bool]  # return True for loop-break
DisconnectHandler = Callable[[ClientData], None]


class AuxSubcontroller:
    """
    Shared by other subcontrollers for various functionality.
    """

    def __init__(self, data: LoopControlData):
        self.data = data
        self._lock = threading.Lock()
        self._pauses_acked_cv = threading.Condition(self._lock)
        self._pause_set: Set[ClientId] = set()
        self._new_model_subscribers: List[NewModelSubscriber] = []

    def subscribe_to_new_model_announcements(self, subscriber: NewModelSubscriber):
        with self._lock:
            self._new_model_subscribers.append(subscriber)

    def broadcast_new_model(self, generation: Generation):
        with self._lock:
            subscribers = list(self._new_model_subscribers)
        for subscriber in subscribers:
            subscriber.handle_new_model(generation)

    def handle_disconnect(self, client_data: ClientData):
        logger.info(f'Handling disconnect: {client_data}...')
        self.data.remove_client(client_data.client_id)
        self.data.close_db_conns(threading.get_ident())
        client_data.socket.close()
        with self._lock:
            self._pause_set.discard(client_data.client_id)

    def send_asset(self, tgt: str, client_data: ClientData):
        all_assets = [self.data.binary_asset] + self.data.extra_assets
        requested_assets = [a for a in all_assets if a.tgt_path == tgt]
        if len(requested_assets) != 1:
            raise ValueError(f'Invalid asset request: {tgt}')

        asset = requested_assets[0]
        src = asset.src_path
        client_data.socket.send_file(src)

    def accept_client(self) -> ClientData:
        conn = self.data.clients_db_conn_pool.get_connection()
        client_socket, addr = self.data.server_socket.accept()
        ip_address, port = addr

        msg = recv_json(client_socket)
        logger.debug(f'Received json message: {msg}')
        assert msg['type'] == 'handshake', f'Expected handshake from client, got {msg}'
        role = msg['role']
        reserved_id = msg.get('reserved_client_id', None)
        client_type = ClientType(role)

        start_timestamp = msg['start_timestamp']
        cuda_device = msg.get('cuda_device', '')
        client_id = None

        with self._lock:
            cursor = conn.cursor()
            if reserved_id is not None:
                # validate that reserved_id is in the clients table
                cursor.execute('SELECT ip_address, port, role, start_timestamp, cuda_device '
                               'FROM clients WHERE id = ?', (reserved_id,))
                row = cursor.fetchone()
                if row is None:
                    logger.warn(
                        f'Client requested reserved id {reserved_id} but it was never reserved')
                elif any([x is not None for x in row]):
                    logger.warn(
                        f'Client requested reserved id {reserved_id} but it is already in use')
                else:
                    cursor.execute("""
                        UPDATE clients
                        SET ip_address = ?, port = ?, role = ?, start_timestamp = ?, cuda_device = ?
                        WHERE id = ?
                    """, (ip_address, port, role, start_timestamp, cuda_device, reserved_id))
                    client_id = reserved_id

            if client_id is None:
                cursor.execute('INSERT INTO clients (ip_address, port, role, start_timestamp, cuda_device) VALUES (?, ?, ?, ?, ?)',
                               (ip_address, port, role, start_timestamp, cuda_device)
                               )
                client_id = cursor.lastrowid

            conn.commit()

        client_data = ClientData(
            client_type, client_id, Socket(client_socket), start_timestamp, cuda_device)

        self.data.add_client(client_data)

        logger.info(f'Accepted client: {client_data}')
        return client_data

    def mark_as_paused(self, client_id: ClientId):
        with self._lock:
            self._pause_set.add(client_id)

    def handle_pause_ack(self, client_data: ClientData):
        with self._lock:
            self._pause_set.discard(client_data.client_id)
            if not self._pause_set:
                self._pauses_acked_cv.notify_all()

    def pause(self, clients: List[ClientData]):
        logger.debug(f'Pausing {len(clients)} clients...')
        logger.debug(f'Clients: {list(map(str, clients))}')
        if not clients:
            return
        data = {'type': 'pause'}

        for client in clients:
            try:
                self.mark_as_paused(client.client_id)
                client.socket.send_json(data)
            except SocketSendException:
                logger.warn(f'Error sending pause to {client}, ignoring...')
                self.handle_disconnect(client)

    def wait_for_pause_acks(self):
        logger.debug(f'Waiting for pause acks...')
        with self._pauses_acked_cv:
            self._pauses_acked_cv.wait_for(lambda: not self._pause_set)
        logger.debug('All pause acks received!')

    def pause_shared_gpu_workers(self):
        self_play_workers = self.data.get_clients(ClientType.SELF_PLAY_WORKER, shared_gpu=True)
        ratings_workers = self.data.get_clients(ClientType.RATINGS_WORKER, shared_gpu=True)
        workers = self_play_workers + ratings_workers
        self.pause(workers)
        self.wait_for_pause_acks()

    def handle_new_model(self, gen: Generation):
        self.reload_weights(self.data.get_clients(ClientType.SELF_PLAY_WORKER), gen)
        self.unpause(self.data.get_clients(ClientType.RATINGS_WORKER, shared_gpu=True))
        self.broadcast_new_model(gen)

    def reload_weights(self, clients: List[ClientData], gen: int):
        if not clients:
            return

        logger.debug(f'Issuing reload weights (gen={gen})...')

        data = {
            'type': 'reload-weights',
            'generation': gen,
        }

        model_filename = self.data.organizer.get_model_filename(gen)
        for client in clients:
            with client.socket.send_mutex():
                send_json(client.socket.native_socket(), data)
                send_file(client.socket.native_socket(), model_filename)

        logger.debug('Reload weights complete!')

    def unpause(self, clients: List[ClientData]):
        if not clients:
            return

        logger.debug(f'Unpausing {len(clients)} clients...')
        logger.debug(f'Clients: {list(map(str, clients))}')

        data = {
            'type': 'unpause',
        }

        for client in clients:
            client.socket.send_json(data)

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

    def launch_recv_loop(self, msg_handler: MsgHandler, client_data: ClientData, thread_name: str,
                         disconnect_handler: Optional[DisconnectHandler] = None):
        """
        Launches a daemon thread that loops, receiving json messages from the client and calling
        msg_handler(client_data, msg) for each message.

        Catches and logs client-disconnection exceptions. Includes a full stack-trace for the more
        uncommon case where the disconnect is detected during a send operation, and a shorter
        single-line message for the more common case where the disconnect is detected during a recv
        operation.

        Signals an error for other types of exceptions; this will cause the entire process to shut
        down.
        """
        threading.Thread(target=self._launch_recv_loop_inner, name=thread_name,
                         args=(msg_handler, disconnect_handler, client_data, thread_name),
                         daemon=True).start()

    def _launch_recv_loop_inner(
            self, msg_handler: MsgHandler, disconnect_handler: DisconnectHandler,
            client_data: ClientData, thread_name: str):
        try:
            while True:
                msg = client_data.socket.recv_json()
                if msg_handler(client_data, msg):
                    break
        except SocketRecvException:
            logger.warn(
                f'Encountered SocketRecvException in {thread_name} (client={client_data}):')
        except SocketSendException:
            logger.warn(
                f'Encountered SocketSendException in {thread_name} (client={client_data}):',
                exc_info=True)
        except:
            logger.error(
                f'Unexpected error in {thread_name} (client={client_data}):', exc_info=True)
            self.data.signal_error()
        finally:
            try:
                if disconnect_handler is not None:
                    disconnect_handler(client_data)
                self.handle_disconnect(client_data)
            except:
                logger.error(
                    f'Error handling disconnect in {thread_name} (client={client_data}):',
                    exc_info=True)
                self.data.signal_error()