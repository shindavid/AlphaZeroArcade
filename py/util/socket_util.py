from dataclasses import dataclass
import json
import logging
import os
import socket
import tempfile
import threading
from typing import Any, Dict, Optional, Union


@dataclass
class EncodedJson:
    header: bytes  # 4-byte big-endian integer specifying the length of the payload
    payload: bytes


class SocketException(Exception):
    """
    The socket module raises a socket.error (an alias for OSError) for failed send/recv operations.
    This choice can make it difficult for the caller to differentiate between a socket error and a
    non-socket OSError.

    Therefore, whenever we catch a socket.error during socket.{send*, recv*}() calls, we raise a
    SocketException instead.

    A typical expected occurrence of this exception is when the socket is closed by the peer.
    """
    pass


class SocketRecvException(SocketException):
    """
    Raised when a socket.recv*() call fails. This is a subclass of SocketException, and is used to
    differentiate between socket.send*() and socket.recv*() calls.
    """
    pass


class SocketSendException(SocketException):
    """
    Raised when a socket.send*() call fails. This is a subclass of SocketException, and is used to
    differentiate between socket.send*() and socket.recv*() calls.
    """
    pass


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
    - SocketRecvException if the socket was closed by peer.
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
                raise SocketRecvException('Socket gracefully closed by peer')
            data.extend(packet)
    except socket.error:
        raise SocketRecvException(
            'socket.recv() failure during recvall() - socket likely closed by peer')
    finally:
        sock.settimeout(None)

    return data


def recv_json(sock: socket.socket,
              timeout: Optional[float] = None,
              log_level=logging.DEBUG) -> Optional[JsonDict]:
    """
    Extracts a json message from the socket and returns it as a dict. Assumes that the json message
    is prepended by a 4-byte big-endian integer specifying the length of the message.

    Calls recvall() under the hood - see recvall() documentation for details on possible exceptions.
    """

    data = recvall(sock, 4, timeout=timeout)
    length = int.from_bytes(data, byteorder='big')

    data = recvall(sock, length, timeout=timeout)
    msg = json.loads(data.decode())
    return msg


def send_json(sock: socket.socket, data: JsonData):
    """
    Sends json data to the socket. This takes the form of a 4-byte big-endian integer specifying
    the length of the message, followed by the encoded message.

    You have the option of passing a raw JsonDict, or an EncodedJson object. The former is
    typically more convenient. The latter can be more efficient in certain cases (e.g., you intend
    to send the same message to multiple sockets, or you want to do the encoding outside of a
    mutex).

    Logs to debug.

    Raises:
    - SocketSendException if the socket was closed by peer.
    """
    if not isinstance(data, EncodedJson):
        data = encode_json(data)

    try:
        sock.sendall(data.header)
        sock.sendall(data.payload)
    except socket.error:
        raise SocketSendException(
            'socket.sendall() failure during send_json() - socket likely closed by peer')


def recv_file(sock: socket.socket, filename: str) -> bytes:
    """
    Receives a file from the socket. This assumes that the file is prepended by a 4-byte big-endian
    integer specifying the length of the file and a 1-byte bool specifying whether the file is
    executable. The file is written to the given filename.

    Calls recvall() under the hood - see recvall() documentation for details on possible exceptions.
    """
    data = recvall(sock, 4)
    length = int.from_bytes(data, byteorder='big')

    data = recvall(sock, 1)
    executable = bool(int.from_bytes(data, byteorder='big'))

    data = recvall(sock, length)
    tmp_filename = tempfile.mktemp()
    with open(tmp_filename, 'wb') as f:
        f.write(data)
    os.rename(tmp_filename, filename)

    if executable:
        os.chmod(filename, 0o755)


def send_file(sock: socket.socket, filename: str):
    """
    Sends the contents of the file to the socket. This takes the form of a 4-byte big-endian integer
    specifying the length of the file, followed by a 1-byte bool specifying whether the file is
    executable, followed by the file itself.

    Raises:
    - SocketSendException if the socket was closed by peer.
    """
    with open(filename, 'rb') as f:
        data = f.read()

    n_bytes = len(data)
    header = n_bytes.to_bytes(4, byteorder='big')
    executable = os.access(filename, os.X_OK)

    try:
        sock.sendall(header)
        sock.sendall(executable.to_bytes(1, byteorder='big'))
        sock.sendall(data)
    except socket.error:
        raise SocketSendException(
            'socket.sendall() failure during send_file() - socket likely closed by peer')


class Socket:
    """
    A wrapper around a socket.socket object that has read/write mutexes.
    """
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._recv_mutex = threading.Lock()
        self._send_mutex = threading.Lock()

    def getsockname(self):
        return self._sock.getsockname()

    def close(self):
        self._sock.close()

    def recv_mutex(self):
        """
        Returns the recv mutex. This can be used in conjunction with standalone socket_util.recv_*()
        calls if you want to hold the mutex across multiple calls.
        """
        return self._recv_mutex

    def send_mutex(self):
        """
        Returns the send mutex. This can be used in conjunction with standalone socket_util.send_*()
        calls if you want to hold the mutex across multiple calls.
        """
        return self._send_mutex

    def recv_json(self, timeout: Optional[float] = None) -> Optional[JsonDict]:
        """
        Mutex-protected call to socket_util.recv_json().
        """
        with self._recv_mutex:
            return recv_json(self._sock, timeout=timeout)

    def send_json(self, data: JsonData):
        """
        Mutex-protected call to socket_util.send_json().
        """
        with self._send_mutex:
            send_json(self._sock, data)

    def recv_file(self, filename: str):
        """
        Mutex-protected call to socket_util.recv_file().
        """
        with self._recv_mutex:
            recv_file(self._sock, filename)

    def send_file(self, filename: str):
        """
        Mutex-protected call to socket_util.send_file().
        """
        with self._send_mutex:
            send_file(self._sock, filename)
