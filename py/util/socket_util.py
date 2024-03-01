from util.logging_util import get_logger

from dataclasses import dataclass
import json
import logging
import os
import socket
from typing import Any, Dict, Optional, Union


logger = get_logger()

@dataclass
class EncodedJson:
    header: bytes  # 4-byte big-endian integer specifying the length of the payload
    payload: bytes


JsonDict = Dict[str, Any]
JsonData = Union[JsonDict, EncodedJson]


def encode_json(data: JsonDict) -> EncodedJson:
    payload = json.dumps(data).encode()
    header = len(payload).to_bytes(4, byteorder='big')
    return EncodedJson(header, payload)


def recvall(sock: socket.socket, n: int, timeout: Optional[float]=None) -> bytearray:
    """Receive exactly n bytes from the socket, looping internally if necessary.

    Args:
    - sock: the socket object to read from. Must be in blocking mode
    - n: the number of bytes to read
    - timeout: the timeout in seconds, or None for blocking mode (default)

    Raises:
    - ValueError if n <= 0, or if socket is in non-blocking mode, or if timeout is not positive
    - socket.timeout if timeout is specified and the socket operation times out.
    - ConnectionError if the socket was gracefully closed by peer.
    - ConnectionResetError if the socket was forcefully reset by peer.
    - OSError if there was an OS-related error.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if sock.gettimeout() is not None:
        raise ValueError("Socket must be in blocking mode")

    if timeout is not None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        sock.settimeout(timeout)

    data = bytearray()
    try:
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError('Socket gracefully closed by peer')
            data.extend(packet)
    finally:
        sock.settimeout(None)

    return data


def recv_json(sock: socket.socket, timeout: Optional[float] = None) -> Optional[JsonDict]:
    """
    Extracts a json message from the socket and returns it as a dict. Assumes that the json message
    is prepended by a 4-byte big-endian integer specifying the length of the message.

    If a ConnectionError is raised, this means the socket is closed. Catches the exception and
    returns None in this case.

    Other exceptions are not caught. See recvall() for details on possible exceptions.
    """

    data = recvall(sock, 4, timeout=timeout)
    length = int.from_bytes(data, byteorder='big')

    data = recvall(sock, length, timeout=timeout)
    msg = json.loads(data.decode())
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'Received json message: {msg}')
    return msg


def send_json(sock: socket.socket, data: JsonData):
    """
    Sends json data to the socket. This takes the form of a 4-byte big-endian integer specifying
    the length of the message, followed by the encoded message.

    You have the option of passing a raw JsonDict, or an EncodedJson object. The former is
    typically more convenient. The latter can be more efficient in certain cases (e.g., you intend
    to send the same message to multiple sockets, or you want to do the encoding outside of a
    mutex).

    Does not do any exception handling; all exceptions raised by sock.send*() are propagated.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'Sending json message: {data}')
    if not isinstance(data, EncodedJson):
        data = encode_json(data)

    sock.sendall(data.header)
    sock.sendall(data.payload)


def recv_file(sock: socket.socket, filename: str) -> bytes:
    """
    Receives a file from the socket. This assumes that the file is prepended by a 4-byte big-endian
    integer specifying the length of the file and a 1-byte bool specifying whether the file is
    executable. The file is written to the given filename.
    """
    data = recvall(sock, 4)
    length = int.from_bytes(data, byteorder='big')

    data = recvall(sock, 1)
    executable = bool(int.from_bytes(data, byteorder='big'))

    data = recvall(sock, length)
    with open(filename, 'wb') as f:
        f.write(data)

    if executable:
        os.chmod(filename, 0o755)

    logger.debug(f'Received file {filename} of size {length} bytes (executable: {executable})')


def send_file(sock: socket.socket, filename: str):
    """
    Sends the contents of the file to the socket. This takes the form of a 4-byte big-endian integer
    specifying the length of the file, followed by a 1-byte bool specifying whether the file is
    executable, followed by the file itself.

    Does not do any exception handling; all exceptions raised by sock.send*() are propagated.
    """
    with open(filename, 'rb') as f:
        data = f.read()

    n_bytes = len(data)
    header = n_bytes.to_bytes(4, byteorder='big')
    executable = os.access(filename, os.X_OK)
    logger.debug(f'Sending file {filename} of size {n_bytes} bytes')
    sock.sendall(header)
    sock.sendall(executable.to_bytes(1, byteorder='big'))
    sock.sendall(data)


def is_port_open(host, port, timeout_sec=1):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_sec)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, socket.error):
            return False
